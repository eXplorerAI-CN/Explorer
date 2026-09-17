# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from model_training.net.transformer import ArtifixerTransformer
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase


class ArtifixerPipeline(ArtifixerPipelineBase):

    def __init__(
        self,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        transformer: WanTransformer3DModel,
        tokenizer: AutoTokenizer | None,
        text_encoder: UMT5EncoderModel | None,
        default_negative_prompt_path: Path,
        frames_per_block: int | None,
        gradient_checkpointing: bool,
        checkpoint_every_n_blocks: int,
        attention_backend: str | None = None,
    ):
        super().__init__(vae, tokenizer, text_encoder)

        self.default_negative_prompt = nn.Buffer(
            torch.load(default_negative_prompt_path, weights_only=True), persistent=False
        )
        self.scheduler = scheduler

        transformer.train()
        transformer.requires_grad_(True)
        if attention_backend is not None:
            transformer.set_attention_backend(attention_backend)

        self.transformer = ArtifixerTransformer(
            transformer,
            frames_per_block=frames_per_block,
            local_attn_size=None,
            sink_size=0,
            vae_scale_factor_spatial=self.vae.config.scale_factor_spatial,
            vae_scale_factor_temporal=self.vae.config.scale_factor_temporal,
            gradient_checkpointing=gradient_checkpointing,
            checkpoint_every_n_blocks=checkpoint_every_n_blocks,
        )

    @torch.no_grad()
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
    ) -> torch.Tensor:
        condition = self.encode_video_frames(rendered_rgb)

        batch_size = rendered_rgb.shape[0]
        if neighbors is not None:
            if neighbors.shape[0] != batch_size:
                raise ValueError("`neighbors` batch size has to be the same as `degraded_renderings` batch size")
            neighbors_condition = self.encode_neighbors(neighbors, max_neighbors_per_encode)
        else:
            neighbors_condition = None

        latents = self.prepare_latents(condition, rendered_opacity, True)

        if isinstance(prompt, torch.Tensor):
            if prompt.ndim != 3:
                raise ValueError("`prompt` has to be a 3D tensor")
            if prompt.shape[0] != batch_size:
                raise ValueError("`prompt` batch size has to be the same as `degraded_renderings` batch size")
            prompt_embeds = prompt
        else:
            prompt_embeds = self.get_t5_prompt_embeds(prompt).expand(batch_size, -1, -1)

        if negative_prompt is None:
            negative_prompt_embeds = self.default_negative_prompt.expand(batch_size, -1, -1)
        elif isinstance(negative_prompt, torch.Tensor):
            if negative_prompt.ndim != 3:
                raise ValueError("`negative_prompt` has to be a 3D tensor")
            if negative_prompt.shape[0] != batch_size:
                raise ValueError("`negative_prompt` batch size has to be the same as `degraded_renderings` batch size")
            negative_prompt_embeds = negative_prompt
        else:
            negative_prompt_embeds = self.get_t5_prompt_embeds(negative_prompt).expand(batch_size, -1, -1)

        self.scheduler.set_timesteps(num_inference_steps, device=self.vae.device)
        timesteps = self.scheduler.timesteps

        for t in tqdm(timesteps, disable=not show_progress, leave=progress_bar_leave):
            timestep = t.expand(latents.shape[0])

            common_kwargs = dict(
                hidden_states=latents,
                timestep=timestep.to(latents.dtype),
                neighbor_hidden_states=neighbors_condition,
                opacity=rendered_opacity,
                camera_rays=camera_rays,
                w2cs=w2cs,
                neighbor_w2cs=neighbor_w2cs,
                Ks=Ks,
                neighbor_Ks=neighbor_Ks,
                return_dict=False,
            )

            noise_pred = self.transformer(encoder_hidden_states=prompt_embeds, **common_kwargs)[0]

            if text_guidance_scale > 1:
                noise_uncond = self.transformer(encoder_hidden_states=negative_prompt_embeds, **common_kwargs)[0]
                noise_pred = noise_uncond + text_guidance_scale * (noise_pred - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        video = self.latents_to_rgb(latents)
        return self.video_processor.postprocess_video(video, output_type="pt")
