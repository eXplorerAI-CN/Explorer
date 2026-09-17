# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from torch import nn

from model_training.pipeline.pipeline import ArtifixerPipeline
from model_training.schedulers.flow_match import FlowMatchScheduler
from model_training.trainers.trainer_base import TrainerBase


class Trainer(TrainerBase):

    pipe: ArtifixerPipeline

    def __init__(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_dataloader: torch.utils.data.DataLoader,
        val_datasets: dict[int, torch.utils.data.Dataset],
        pipe: ArtifixerPipeline,
        frames_per_block: int | None,
        step_offset: int = 0,
    ):
        super().__init__(
            args, accelerator, pipe, [optimizer], [scheduler], train_dataloader, val_datasets, True, step_offset
        )
        self.frames_per_block = frames_per_block

        self.train_scheduler = FlowMatchScheduler(
            shift=5, sigma_min=0.0, extra_one_step=True, device=self.pipe.vae.device
        )

    def training_batch(self, data: dict) -> tuple[torch.Tensor, dict]:
        data = self.process_inputs(data)
        batch_size = data["rgb_gt"].shape[0]
        noisy_latent = self.pipe.prepare_latents(data["condition"], data["opacity"], True)
        latent_gt = self.pipe.encode_video_frames(data["rgb_gt"])

        if self.frames_per_block is not None:
            timestep_id = torch.randint(
                0,
                self.train_scheduler.num_train_timesteps,
                (batch_size * latent_gt.shape[2] // self.frames_per_block,),
            )
        else:
            timestep_id = torch.randint(0, self.train_scheduler.num_train_timesteps, (batch_size,))

        timestep = self.train_scheduler.timesteps[timestep_id]

        if self.frames_per_block is not None:
            model_input = self.train_scheduler.add_noise_multi_timestep(latent_gt, noisy_latent, timestep).to(
                latent_gt.dtype
            )
        else:
            model_input = self.train_scheduler.add_noise(latent_gt, noisy_latent, timestep).to(latent_gt.dtype)

        with torch.no_grad():
            training_target = self.train_scheduler.training_target(latent_gt, noisy_latent)

        noise_pred = self.pipe.transformer(
            hidden_states=model_input,
            timestep=timestep.to(model_input.dtype),
            encoder_hidden_states=data["encoded_prompt"],
            neighbor_hidden_states=data["neighbors_condition"],
            ignore_neighbors=data["ignore_neighbors"],
            opacity=data["opacity"],
            camera_rays=data["camera_rays"],
            w2cs=data["w2cs"],
            neighbor_w2cs=data["neighbor_w2cs"],
            Ks=data["Ks"],
            neighbor_Ks=data["neighbor_Ks"],
            return_dict=False,
        )[0]

        mse_loss = torch.nn.functional.mse_loss(
            noise_pred.float(),
            training_target.float(),
            reduction="none" if self.frames_per_block is not None else "mean",
        )
        metrics = {
            "mse_loss": mse_loss.mean().item() if self.frames_per_block is not None else mse_loss.item(),
        }
        loss = mse_loss

        if self.frames_per_block is not None:
            weights = self.train_scheduler.training_weight_multi_timestep(timestep)
            weights = weights.repeat_interleave(loss.shape[0] * loss.shape[2] // weights.shape[0]).view(
                loss.shape[0], 1, loss.shape[2], 1, 1
            )
            loss = (loss * weights).mean()
        else:
            loss = loss * self.train_scheduler.training_weight(timestep)

        return loss, metrics

    @torch.no_grad()
    def validation_batches(
        self,
        val_datasets: dict[int, torch.utils.data.Dataset],
        validation_index: int,
        save_dir: Path,
    ) -> None:
        self.pipe.transformer.eval()
        self._run_and_save_validation_tasks(val_datasets, validation_index, 50, save_dir)
        self.pipe.transformer.train()

    def to_accumulate(self) -> list[nn.Module]:
        return [self.pipe.transformer]

    def on_before_optimizer_step(self) -> None:
        # https://github.com/huggingface/accelerate/issues/3789
        self.accelerator.unscale_gradients()
        torch.nn.utils.clip_grad_norm_(self.pipe.transformer.parameters(), self.args.max_grad_norm)

    def on_before_zero_grad(self) -> None:
        pass
