# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn
from torch.distributed.fsdp import FSDPModule

from model_training.net.transformer import ArtifixerTransformer
from model_training.pipeline.kv_cache_pipeline import ArtifixerKvCachePipeline
from model_training.trainers.trainer_base import TrainerBase
from model_training.utils.ema import DTensorFastEmaModelUpdater


class DMD(TrainerBase):

    pipe: ArtifixerKvCachePipeline

    def __init__(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        optimizers: list[torch.optim.Optimizer],
        schedulers: list[torch.optim.lr_scheduler.LRScheduler],
        train_dataloader: torch.utils.data.DataLoader,
        val_datasets: dict[int, torch.utils.data.Dataset],
        pipe: ArtifixerKvCachePipeline,
        real_score: ArtifixerTransformer,
        fake_score: ArtifixerTransformer,
        ema: ArtifixerTransformer | None,
        ema_worker: DTensorFastEmaModelUpdater | None,
        step_offset: int = 0,
    ):
        super().__init__(
            args, accelerator, pipe, optimizers, schedulers, train_dataloader, val_datasets, False, step_offset
        )
        self.real_score = real_score
        self.fake_score = fake_score
        self.ema = ema
        self.ema_worker = ema_worker
        self.negative_prompt = torch.load(args.default_negative_prompt_path, weights_only=True).to(pipe.vae.device)

        self.min_step = int(0.02 * self.pipe.scheduler.num_train_timesteps)
        self.max_step = int(0.98 * self.pipe.scheduler.num_train_timesteps)

    def training_batch(self, data: dict) -> tuple[torch.Tensor, dict]:
        # Process inputs once so critic and generator see the same dropout masks
        # and conditioning — matching the official DMD2 implementation where
        # guidance_data_dict is shared between turns.
        data = self.process_inputs(data)

        # Generate x_fake once; shared between critic (detached) and generator.
        prediction = self.pipe.generate_samples_from_batch(
            data["condition"],
            data["opacity"],
            data["neighbors_condition"],
            data["camera_rays"],
            data["w2cs"],
            data["neighbor_w2cs"],
            data["Ks"],
            data["neighbor_Ks"],
            data["encoded_prompt"],
            self.args.num_inference_steps,
            True,
            ignore_neighbors=data["ignore_neighbors"],
        )

        critic_loss = self.critic_loss(data, prediction.detach())
        metrics = {"critic_loss": critic_loss.item()}
        loss = critic_loss

        if self.is_student_phase():
            generator_loss = self.generator_loss(data, prediction)
            loss += generator_loss
            metrics["generator_loss"] = generator_loss.item()

        return loss, metrics

    def generator_loss(self, data: dict, prediction: torch.Tensor):
        """Compute the DMD distribution matching loss (DMD2 paper eq. 7-8)."""
        with torch.no_grad():
            timestep = self.get_timestep(prediction.shape[0])
            noise = self.pipe.prepare_latents(data["condition"], data["opacity"], True)
            noisy_latent = self.pipe.scheduler.add_noise(prediction, noise, timestep).detach()

            score_kwargs = dict(
                hidden_states=noisy_latent,
                timestep=timestep.to(noisy_latent.dtype),
                neighbor_hidden_states=data["neighbors_condition"],
                ignore_neighbors=data["ignore_neighbors"],
                opacity=data["opacity"],
                camera_rays=data["camera_rays"],
                w2cs=data["w2cs"],
                neighbor_w2cs=data["neighbor_w2cs"],
                Ks=data["Ks"],
                neighbor_Ks=data["neighbor_Ks"],
                return_dict=False,
            )

            fake_noise_pred = self.fake_score(encoder_hidden_states=data["encoded_prompt"], **score_kwargs)[0]
            fake_predicted_x0 = self.pipe.scheduler.step(fake_noise_pred, timestep, noisy_latent, to_final=True)

            real_noise_pred = self.real_score(encoder_hidden_states=data["encoded_prompt"], **score_kwargs)[0]
            real_predicted_x0 = self.pipe.scheduler.step(real_noise_pred, timestep, noisy_latent, to_final=True)

            if self.args.text_guidance_scale > 1:
                neg_prompt = self.negative_prompt.expand(prediction.shape[0], -1, -1)
                fake_noise_pred_uncond_text = self.fake_score(encoder_hidden_states=neg_prompt, **score_kwargs)[0]
                fake_predicted_x0_uncond_text = self.pipe.scheduler.step(
                    fake_noise_pred_uncond_text, timestep, noisy_latent, to_final=True
                )
                fake_predicted_x0 = (
                    fake_predicted_x0_uncond_text
                    + (fake_predicted_x0 - fake_predicted_x0_uncond_text) * self.args.text_guidance_scale
                )

                real_noise_pred_uncond_text = self.real_score(encoder_hidden_states=neg_prompt, **score_kwargs)[0]
                real_predicted_x0_uncond_text = self.pipe.scheduler.step(
                    real_noise_pred_uncond_text, timestep, noisy_latent, to_final=True
                )
                real_predicted_x0 = (
                    real_predicted_x0_uncond_text
                    + (real_predicted_x0 - real_predicted_x0_uncond_text) * self.args.text_guidance_scale
                )

            # DMD gradient (eq. 7) with normalization (eq. 8).
            grad = fake_predicted_x0 - real_predicted_x0
            p_real = prediction - real_predicted_x0
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = torch.nan_to_num(grad / normalizer)

        return 0.5 * F.mse_loss(prediction.double(), (prediction.double() - grad.double()).detach(), reduction="mean")

    def critic_loss(self, data: dict, prediction: torch.Tensor):
        """Train the fake score network on the generator's detached output (DMD2 Sec 4.5)."""
        timestep = self.get_timestep(prediction.shape[0])
        noise = self.pipe.prepare_latents(data["condition"], data["opacity"], True)
        noisy_latent = self.pipe.scheduler.add_noise(prediction, noise, timestep)
        score_kwargs = dict(
            hidden_states=noisy_latent,
            timestep=timestep.to(noisy_latent.dtype),
            neighbor_hidden_states=data["neighbors_condition"],
            ignore_neighbors=data["ignore_neighbors"],
            opacity=data["opacity"],
            camera_rays=data["camera_rays"],
            w2cs=data["w2cs"],
            neighbor_w2cs=data["neighbor_w2cs"],
            Ks=data["Ks"],
            neighbor_Ks=data["neighbor_Ks"],
            return_dict=False,
        )
        fake_noise_pred = self.fake_score(encoder_hidden_states=data["encoded_prompt"], **score_kwargs)[0]

        with torch.no_grad():
            training_target = self.pipe.scheduler.training_target(prediction, noise)
        return F.mse_loss(fake_noise_pred.float(), training_target.float())

    @torch.no_grad()
    def validation_batches(
        self,
        val_datasets: dict[int, torch.utils.data.Dataset],
        validation_index: int,
        save_dir: Path,
    ) -> None:
        if self.ema_worker is not None and self.get_step() >= self.args.ema_start_step:
            # https://github.com/pytorch/pytorch/issues/144289
            for module in self.pipe.transformer.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.ema_worker.cache(self.pipe.transformer.parameters())
            self.ema_worker.copy_to(src_model=self.ema, tgt_model=self.pipe.transformer)

        self.pipe.transformer.eval()
        self._run_and_save_validation_tasks(
            val_datasets,
            validation_index,
            self.args.num_inference_steps,
            save_dir,
        )

        if self.ema_worker is not None and self.get_step() >= self.args.ema_start_step:
            for module in self.pipe.transformer.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.ema_worker.restore(self.pipe.transformer.parameters())

        self.pipe.transformer.train()

    def to_accumulate(self) -> list[nn.Module]:
        return [self.pipe.transformer, self.fake_score]

    def on_before_optimizer_step(self) -> None:
        # https://github.com/huggingface/accelerate/issues/3789
        self.accelerator.unscale_gradients()
        torch.nn.utils.clip_grad_norm_(self.pipe.transformer.parameters(), self.args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.fake_score.parameters(), self.args.max_grad_norm)

    def on_before_zero_grad(self) -> None:
        if not self.is_student_phase():
            return

        if self.accelerator.sync_gradients and self.ema_worker is not None:
            self.ema_worker.update_average(
                self.pipe.transformer,
                self.ema,
                beta=self.args.ema_weight if (self.get_step() >= self.args.ema_start_step) else 0,
            )

    def is_student_phase(self) -> bool:
        return (self.get_step() - 1) % self.args.dfake_gen_update_ratio == 0

    def get_timestep(self, batch_size: int) -> torch.Tensor:
        timestep = torch.randint(
            0, self.pipe.scheduler.num_train_timesteps, (batch_size,), device=self.pipe.vae.device, dtype=torch.long
        )

        if self.args.timestep_shift > 1:
            timestep = (
                self.args.timestep_shift
                * (timestep / self.pipe.scheduler.num_train_timesteps)
                / (1 + (self.args.timestep_shift - 1) * (timestep / self.pipe.scheduler.num_train_timesteps))
                * self.pipe.scheduler.num_train_timesteps
            )

        return timestep.clamp(self.min_step, self.max_step)
