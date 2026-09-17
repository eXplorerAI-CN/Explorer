# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel

from model_training.constants import MAX_SEQUENCE_LENGTH


class ArtifixerPipelineBase(nn.Module, ABC):

    def __init__(
        self,
        vae: AutoencoderKLWan,
        tokenizer: AutoTokenizer | None,
        text_encoder: UMT5EncoderModel | None,
    ):
        super().__init__()

        self.tokenizer = tokenizer

        if text_encoder is not None:
            assert tokenizer is not None, "`tokenizer` is required when `text_encoder` is provided"
            self.text_encoder = text_encoder
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)
        else:
            self.text_encoder = None

        self.vae = vae
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.config.scale_factor_spatial)

        self.latents_mean = torch.nn.Buffer(
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype),
            persistent=False,
        )
        self.latents_std = torch.nn.Buffer(
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype),
            persistent=False,
        )

    def cp_frame_divisor(self, total_latent_frames: int) -> int:
        """Return the GCD of all frame counts that CP group size must divide.

        The default is ``total_latent_frames`` (the whole video is processed in
        one transformer call).  Subclasses that process chunks (e.g. KV-cache
        pipeline) should override to reflect per-chunk and cache constraints.

        Example::

            # 21 latent frames -> returns 21
            # Valid CP sizes are divisors of 21: {1, 3, 7, 21}
            pipe.cp_frame_divisor(21)  # 21
        """
        return total_latent_frames

    @abstractmethod
    def forward_inference(
        self,
        rendered_rgb: torch.Tensor,
        rendered_opacity: torch.Tensor,
        neighbors: torch.Tensor | None,
        camera_rays: torch.Tensor,
        w2cs: torch.Tensor,
        neighbor_w2cs: torch.Tensor | None,
        Ks: torch.Tensor,
        neighbor_Ks: torch.Tensor | None,
        prompt: str | torch.Tensor,
        negative_prompt: str | torch.Tensor | None = None,
        num_inference_steps: int = 50,
        text_guidance_scale: float = 5.0,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        max_neighbors_per_encode: int | None = None,
    ) -> torch.Tensor: ...

    def rgb_to_latents(self, rgb: torch.Tensor) -> torch.Tensor:
        latents = retrieve_latents(self.vae.encode(rgb.to(self.vae.dtype)), sample_mode="argmax")
        return (latents - self.latents_mean) * self.latents_std

    def latents_to_rgb(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.vae.dtype)
        latents = latents / self.latents_std + self.latents_mean
        return self.vae.decode(latents, return_dict=False)[0]

    def prepare_latents(self, condition: torch.Tensor, opacity: torch.Tensor, is_first_chunk: bool) -> torch.Tensor:
        opacity_for_mixing = (
            torch.cat([opacity[:, :1].repeat_interleave(3, dim=1), opacity], dim=1).unsqueeze(1)
            if is_first_chunk
            else opacity
        )
        opacity_for_mixing = torch.nn.functional.max_pool3d(
            opacity_for_mixing,
            (
                self.vae.config.scale_factor_temporal,
                self.vae.config.scale_factor_spatial,
                self.vae.config.scale_factor_spatial,
            ),
        )
        latents = condition * opacity_for_mixing + torch.randn_like(condition) * (1 - opacity_for_mixing)

        # CP ranks must start with identical latents — torch.randn_like uses
        # each GPU's local RNG, producing different noise per rank.  Broadcast
        # from group rank 0 so all ranks in the CP group share the same noise.
        cp_mesh = self.transformer._cp_mesh
        if cp_mesh is not None:
            src_rank = cp_mesh.mesh.view(-1)[0].item()
            dist.broadcast(latents, src=src_rank, group=cp_mesh.get_group())

        return latents

    def encode_video_frames(self, video_frames: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _, height, width = video_frames.shape
        assert height % 16 == 0, "`height` has to be divisible by 16"
        assert width % 16 == 0, "`width` has to be divisible by 16"
        assert (
            num_frames - 1
        ) % self.vae.config.scale_factor_temporal == 0, (
            f"`num_frames - 1` has to be divisible by {self.vae.config.scale_factor_temporal}"
        )

        video_frames = (
            self.video_processor.preprocess(video_frames.flatten(0, 1))
            .view(batch_size, num_frames, 3, height, width)
            .permute(0, 2, 1, 3, 4)
        )

        return self.rgb_to_latents(video_frames)

    def _encode_neighbors_batch(self, neighbor_frames: torch.Tensor) -> torch.Tensor:
        batch_size, num_neighbors, _, neighbors_height, neighbors_width = neighbor_frames.shape
        assert neighbors_height % 16 == 0, "`neighbors_height` has to be divisible by 16"
        assert neighbors_width % 16 == 0, "`neighbors_width` has to be divisible by 16"

        # Not using temporal compression atm
        neighbors_condition = (
            self.video_processor.preprocess(neighbor_frames.flatten(0, 1))
            .view(-1, 1, 3, neighbors_height, neighbors_width)
            .permute(0, 2, 1, 3, 4)
        )
        neighbors_condition = self.rgb_to_latents(neighbors_condition)
        neighbors_condition = neighbors_condition.squeeze(2)
        neighbors_condition = neighbors_condition.view(batch_size, num_neighbors, *neighbors_condition.shape[1:])
        return neighbors_condition.permute(0, 2, 1, 3, 4)

    def encode_neighbors(
        self, neighbor_frames: torch.Tensor, max_neighbors_per_encode: int | None = None
    ) -> torch.Tensor:
        if max_neighbors_per_encode is None:
            return self._encode_neighbors_batch(neighbor_frames)
        if max_neighbors_per_encode <= 0:
            raise ValueError("`max_neighbors_per_encode` must be positive when set")

        chunks = [
            self._encode_neighbors_batch(neighbor.to(self.vae.device))
            for neighbor in neighbor_frames.split(max_neighbors_per_encode, dim=1)
        ]
        return torch.cat(chunks, dim=2)

    def get_t5_prompt_embeds(self, prompt: str) -> torch.Tensor:
        assert self.tokenizer is not None, "`tokenizer` is required when getting T5 prompt embeddings"
        assert self.text_encoder is not None, "`text_encoder` is required when getting T5 prompt embeddings"

        text_inputs = self.tokenizer(
            prompt_clean(prompt),
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device),
            mask.to(self.text_encoder.device),
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.vae.dtype, device=self.vae.device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(MAX_SEQUENCE_LENGTH - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )

        return prompt_embeds
