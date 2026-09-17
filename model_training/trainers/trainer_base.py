# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gc
import math
import shutil
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from PIL import Image
from torch import nn
from torch.distributed import DeviceMesh
from tqdm import tqdm

from model_training.constants import MAX_SEQUENCE_LENGTH
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase
from model_training.utils.train_utils import (
    barrier_if_distributed,
    checkpoint_is_complete,
    mark_checkpoint_complete,
    tracker_logging_enabled,
    uses_wandb_logging,
)
from model_training.utils.video_io import save_video

_VALIDATION_SYNC_TIMEOUT = timedelta(minutes=30)


@dataclass(frozen=True)
class ValidationTaskSpec:
    key: str
    num_neighbors: int
    use_neighbors: bool = True
    zero_rendered: bool = False
    last_drop_fraction: float | None = None
    zero_prompt: bool = False


@dataclass
class ValidationTask:
    key: str
    rendered_rgb: torch.Tensor
    rendered_opacity: torch.Tensor
    neighbors: torch.Tensor | None
    camera_rays: torch.Tensor
    w2cs: torch.Tensor
    neighbor_w2cs: torch.Tensor | None
    Ks: torch.Tensor
    neighbor_Ks: torch.Tensor | None
    prompt: torch.Tensor
    num_inference_steps: int
    rgb_gt: torch.Tensor


class TrainerBase(ABC):

    def __init__(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        pipe: ArtifixerPipelineBase,
        optimizers: list[torch.optim.Optimizer],
        schedulers: list[torch.optim.lr_scheduler.LRScheduler],
        train_dataloader: torch.utils.data.DataLoader,
        val_datasets: dict[int, torch.utils.data.Dataset],
        copy_val_to_device: bool = True,
        step_offset: int = 0,
    ):
        self.args = args
        self.accelerator = accelerator
        self.pipe = pipe
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.train_dataloader = train_dataloader
        self.val_datasets = val_datasets
        self.copy_val_to_device = copy_val_to_device
        self.step_offset = step_offset

    @abstractmethod
    def training_batch(self, data: dict) -> tuple[torch.Tensor, dict]:
        pass

    @abstractmethod
    def validation_batches(
        self,
        val_datasets: dict[int, torch.utils.data.Dataset],
        validation_index: int,
        save_dir: Path,
    ) -> None:
        pass

    @abstractmethod
    def to_accumulate(self) -> list[nn.Module]:
        pass

    @abstractmethod
    def on_before_optimizer_step(self) -> None:
        pass

    @abstractmethod
    def on_before_zero_grad(self) -> None:
        pass

    def _should_checkpoint_for_timeout(self) -> bool:
        if self.args.should_checkpoint_flag is None:
            return False

        # Slurm pre-timeout handlers touch this flag so the job can save a
        # checkpoint before exiting with the requeue-friendly timeout code.
        should_checkpoint = self.args.should_checkpoint_flag.exists()
        if dist.is_initialized():
            # If any rank sees the flag, every rank must enter the checkpoint
            # path together; otherwise distributed collectives can diverge.
            should_checkpoint_tensor = torch.tensor(
                [int(should_checkpoint)],
                device=self.accelerator.device,
                dtype=torch.int32,
            )
            dist.all_reduce(should_checkpoint_tensor, op=dist.ReduceOp.MAX)
            should_checkpoint = bool(should_checkpoint_tensor.item())

        return should_checkpoint

    def _save_state_with_commit_marker(self, step: int) -> None:
        project_dir = Path(self.accelerator.project_dir)
        checkpoint_dir = project_dir / "checkpoints" / f"checkpoint_{step}"
        if checkpoint_is_complete(project_dir, step, self.accelerator):
            return

        if self.accelerator.is_main_process and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        barrier_if_distributed()

        self.accelerator.project_configuration.iteration = step

        local_save_succeeded = True
        caught_error = None
        try:
            self.accelerator.save_state()
        except Exception as e:
            local_save_succeeded = False
            caught_error = e
            self.accelerator.print(
                f"Checkpoint save failed on rank {self.accelerator.process_index} at iteration {step}:\n"
                f"{traceback.format_exc()}"
            )

        if dist.is_initialized():
            save_succeeded_tensor = torch.tensor(
                [int(local_save_succeeded)],
                device=self.accelerator.device,
                dtype=torch.int32,
            )
            dist.all_reduce(save_succeeded_tensor, op=dist.ReduceOp.MIN)
            save_succeeded = bool(save_succeeded_tensor.item())
        else:
            save_succeeded = local_save_succeeded

        if not save_succeeded:
            checkpoint_complete = checkpoint_is_complete(project_dir, step, self.accelerator)
            if self.accelerator.is_main_process and not checkpoint_complete:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            barrier_if_distributed()
            if caught_error is not None:
                raise caught_error
            raise RuntimeError(f"Distributed checkpoint save failed at iteration {step}")

        barrier_if_distributed()
        mark_checkpoint_complete(project_dir, step, self.accelerator)
        barrier_if_distributed()

    def process_inputs(self, data: dict) -> dict[str, torch.Tensor | bool]:
        batch_size, num_frames, _, height, width = data["rgb_gt"].shape
        processed_data = {"rgb_gt": data["rgb_gt"]}

        dropout_probs = torch.rand(3)

        opacity = data["opacity"].to(dtype=self.pipe.vae.dtype)
        if dropout_probs[0] < self.args.dropout_rate:
            condition = self.pipe.encode_video_frames(torch.zeros_like(data["rgb_rendered"]))
            opacity = torch.zeros_like(opacity)
        else:
            frames_to_drop = torch.randint(0, num_frames, (1,))
            if frames_to_drop > 0:
                condition = self.pipe.encode_video_frames(
                    torch.cat(
                        [
                            data["rgb_rendered"][:, : num_frames - frames_to_drop],
                            torch.zeros_like(data["rgb_rendered"][:, -frames_to_drop:]),
                        ],
                        dim=1,
                    )
                )
                opacity[:, -frames_to_drop:] = 0
            else:
                condition = self.pipe.encode_video_frames(data["rgb_rendered"])

        processed_data["condition"] = condition
        processed_data["opacity"] = opacity

        encoded_prompt = data["encoded_prompt"]
        if dropout_probs[1] < self.args.dropout_rate:
            encoded_prompt = torch.zeros_like(encoded_prompt)
        processed_data["encoded_prompt"] = encoded_prompt

        # Training must execute the neighbor branch on every rank: the per-block
        # neighbor KV projections still have gradients even though patch embedding
        # is shared. ``ignore_neighbors`` zeros the contribution after the branch.
        processed_data["neighbors_condition"] = self.pipe.encode_neighbors(data["rgb_neighbors"])
        processed_data["ignore_neighbors"] = dropout_probs[2] < self.args.dropout_rate
        processed_data["neighbor_w2cs"] = data["neighbor_w2cs"]
        processed_data["neighbor_Ks"] = data["neighbor_Ks"]
        processed_data["camera_rays"] = data["camera_rays"].to(dtype=self.pipe.vae.dtype)
        processed_data["w2cs"] = data["w2cs"]
        processed_data["Ks"] = data["Ks"]

        return processed_data

    @staticmethod
    def _build_validation_task_specs(val_datasets: dict[int, torch.utils.data.Dataset]) -> list[ValidationTaskSpec]:
        specs = []

        for num_neighbors in sorted(val_datasets):
            specs.append(ValidationTaskSpec(key=f"default_{num_neighbors}", num_neighbors=num_neighbors))

        fewest_neighbors = min(val_datasets.keys())

        if specs[0].key != f"default_{fewest_neighbors}":
            raise RuntimeError(f"Unexpected first validation task key: {specs[0].key}")

        for last_drop in [0.25, 0.5, 0.75]:
            specs.append(
                ValidationTaskSpec(
                    key=f"last_drop_{last_drop}_{fewest_neighbors}",
                    num_neighbors=fewest_neighbors,
                    last_drop_fraction=last_drop,
                )
            )

        specs.extend(
            [
                ValidationTaskSpec(
                    key=f"text_only_{fewest_neighbors}",
                    num_neighbors=fewest_neighbors,
                    use_neighbors=False,
                    zero_rendered=True,
                ),
                ValidationTaskSpec(
                    key=f"no_rendered_rgb_{fewest_neighbors}",
                    num_neighbors=fewest_neighbors,
                    zero_rendered=True,
                ),
                ValidationTaskSpec(
                    key=f"neighbors_only_{fewest_neighbors}",
                    num_neighbors=fewest_neighbors,
                    zero_rendered=True,
                    zero_prompt=True,
                ),
                ValidationTaskSpec(
                    key=f"no_text_{fewest_neighbors}",
                    num_neighbors=fewest_neighbors,
                    zero_prompt=True,
                ),
                ValidationTaskSpec(
                    key=f"no_neighbors_{fewest_neighbors}",
                    num_neighbors=fewest_neighbors,
                    use_neighbors=False,
                ),
                ValidationTaskSpec(
                    key=f"no_neighbors_no_text_{fewest_neighbors}",
                    num_neighbors=fewest_neighbors,
                    use_neighbors=False,
                    zero_prompt=True,
                ),
            ]
        )

        return specs

    def _load_validation_batch(
        self,
        val_datasets: dict[int, torch.utils.data.Dataset],
        num_neighbors: int,
        validation_index: int,
    ) -> dict:
        data = torch.utils.data.default_collate([val_datasets[num_neighbors][validation_index]])
        if not self.copy_val_to_device:
            return data

        return {
            key: value.to(device=self.accelerator.device) if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }

    def _build_validation_task(
        self,
        val_datasets: dict[int, torch.utils.data.Dataset],
        spec: ValidationTaskSpec,
        validation_index: int,
        num_inference_steps: int,
    ) -> ValidationTask:
        data = self._load_validation_batch(val_datasets, spec.num_neighbors, validation_index)
        rendered_rgb = data["rgb_rendered"]
        rendered_opacity = data["opacity"].to(dtype=self.pipe.vae.dtype)
        prompt = data["encoded_prompt"]

        if spec.zero_rendered:
            rendered_rgb = torch.zeros_like(rendered_rgb)
            rendered_opacity = torch.zeros_like(rendered_opacity)
        elif spec.last_drop_fraction is not None:
            frames_to_drop = int(rendered_rgb.shape[1] * spec.last_drop_fraction)
            rendered_rgb = rendered_rgb.clone()
            rendered_opacity = rendered_opacity.clone()
            if frames_to_drop > 0:
                rendered_rgb[:, -frames_to_drop:] = 0
                rendered_opacity[:, -frames_to_drop:] = 0

        if spec.zero_prompt:
            prompt = torch.zeros_like(prompt)

        return ValidationTask(
            key=spec.key,
            rendered_rgb=rendered_rgb,
            rendered_opacity=rendered_opacity,
            neighbors=data["rgb_neighbors"] if spec.use_neighbors else None,
            camera_rays=data["camera_rays"].to(dtype=self.pipe.vae.dtype),
            w2cs=data["w2cs"],
            neighbor_w2cs=data["neighbor_w2cs"] if spec.use_neighbors else None,
            Ks=data["Ks"],
            neighbor_Ks=data["neighbor_Ks"] if spec.use_neighbors else None,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            rgb_gt=data["rgb_gt"],
        )

    @staticmethod
    def _choose_cp_group_size(
        cp_frame_divisor: int,
        world_size: int,
        num_tasks: int,
    ) -> int:
        """Pick CP group size that minimizes validation wall-clock time.

        Tries all divisors of ``cp_frame_divisor`` (so no temporal padding is
        needed) and picks the one that minimizes
        ``ceil(num_tasks / num_groups) / cp_size``.

        Leftover ranks (``world_size % cp_group_size != 0``) become spectators
        that run a minimal dummy forward pass for FSDP synchronization.  Since
        FSDP all-gathers only reconstruct parameters (input-size-independent),
        spectators process a tiny input and never bottleneck CP-active ranks.

        Examples::

            # cp_frame_divisor=21, world_size=9, num_tasks=12
            #   d=1:  groups=9,  rounds=ceil(12/9)=2,   time=2/1=2.00
            #   d=3:  groups=3,  rounds=ceil(12/3)=4,   time=4/3=1.33  <-- best
            #   d=7:  groups=1,  rounds=ceil(12/1)=12,  time=12/7=1.71
            #   d=21: exceeds world_size, skip
            # Returns 3

            # With block-causal (21 frames, frames_per_block=7):
            #   trainer computes cp_frame_div = gcd(21, num_blocks=3) = 3
            #   d=1: groups=9, rounds=2, time=2.00
            #   d=3: groups=3, rounds=4, time=1.33  <-- best
            # Returns 3 (CP=3, each rank gets 1 temporal block)
        """
        if world_size <= 1:
            return 1
        best_size, best_time = 1, float("inf")
        for d in range(1, cp_frame_divisor + 1):
            if cp_frame_divisor % d != 0 or d > world_size:
                continue
            num_groups = world_size // d
            if num_groups == 0:
                continue
            rounds = math.ceil(num_tasks / num_groups)
            time = rounds / d
            if time < best_time:
                best_time = time
                best_size = d
        return best_size

    def _validation_prompt_dim(self) -> int:
        transformer = self.pipe.transformer
        unwrapped_transformer = getattr(transformer, "module", transformer)

        condition_embedder = getattr(unwrapped_transformer, "condition_embedder", None)
        text_embedder = getattr(condition_embedder, "text_embedder", None)
        linear_1 = getattr(text_embedder, "linear_1", None)
        if linear_1 is not None and hasattr(linear_1, "in_features"):
            return int(linear_1.in_features)

        config = getattr(transformer, "config", None)
        text_dim = getattr(config, "text_dim", None)
        if text_dim is not None:
            return int(text_dim)

        blocks = getattr(unwrapped_transformer, "blocks", None)
        if blocks is not None:
            return int(blocks[0].attn2.to_k.in_features)

        raise RuntimeError("Could not infer validation prompt embedding dimension")

    @staticmethod
    def _build_minimal_validation_task(
        key: str,
        num_frames: int,
        num_latent_frames: int,
        prompt_dim: int,
        num_inference_steps: int,
        device: torch.device,
        vae_dtype: torch.dtype,
    ) -> ValidationTask:
        """Build a minimal-resolution task for ranks that only need FSDP sync.

        Inactive validation groups must still call ``forward_inference`` so that
        FSDP all-gathers fire, but their output is discarded.  Keep the **same
        temporal extent** as real tasks (so the KV-cache pipeline produces the
        same number of chunks / transformer calls — required for FSDP sync)
        but uses tiny 16×16 spatial resolution (1 token per frame instead of
        ~1560), making per-layer compute negligible.

        ``vae_dtype`` (typically bf16) is used for opacity and camera_rays to
        match the dtype that ``_build_validation_task`` produces — these
        tensors bypass the VAE and are mixed directly with bf16 latents in
        ``prepare_latents``, so a float32 mismatch would upcast everything.
        """
        min_hw = 16  # divisible by vae_spatial * patch = 8 * 2 = 16
        pixel_dtype = torch.float32  # pixels go through the VAE, which casts
        B = 1

        return ValidationTask(
            key=key,
            rendered_rgb=torch.zeros(B, num_frames, 3, min_hw, min_hw, device=device, dtype=pixel_dtype),
            rendered_opacity=torch.zeros(B, num_frames, min_hw, min_hw, device=device, dtype=vae_dtype),
            neighbors=None,  # None → skips encode_neighbors, PRoPE, and neighbor cross-attn entirely
            # camera_rays from compute_camera_rays are at latent temporal resolution
            # (poses averaged in groups of 4 for VAE temporal compression), so match that.
            camera_rays=torch.zeros(B, num_latent_frames, min_hw, min_hw, 6, device=device, dtype=vae_dtype),
            w2cs=torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, num_latent_frames, -1, -1),
            neighbor_w2cs=None,
            Ks=torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, num_latent_frames, -1, -1),
            neighbor_Ks=None,
            prompt=torch.zeros(B, MAX_SEQUENCE_LENGTH, prompt_dim, device=device, dtype=torch.bfloat16),
            num_inference_steps=num_inference_steps,
            rgb_gt=torch.zeros(B, num_frames, 3, min_hw, min_hw, device=device, dtype=pixel_dtype),
        )

    def _run_and_save_validation_tasks(
        self,
        val_datasets: dict[int, torch.utils.data.Dataset],
        validation_index: int,
        num_inference_steps: int,
        save_dir: Path,
    ) -> None:
        task_specs = self._build_validation_task_specs(val_datasets)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        validation_sync_group = (
            dist.new_group(backend="nccl", timeout=_VALIDATION_SYNC_TIMEOUT) if dist.is_initialized() else None
        )

        try:
            if rank == 0:
                save_dir.mkdir(parents=True, exist_ok=True)

            barrier_if_distributed(validation_sync_group)

            # Hybrid task-parallel + context-parallel validation.
            # GPUs are split into CP groups; each group cooperates on one task via
            # all-gather K/V attention, and different groups run different tasks in
            # parallel.  For block-causal attention (diffusion forcing), CP is still
            # possible but frames_per_rank must be divisible by frames_per_block,
            # which constrains valid CP sizes to divisors of num_blocks.
            vae_temporal = self.pipe.vae.config.scale_factor_temporal
            reference_dataset = val_datasets[task_specs[0].num_neighbors]
            num_latent_frames = (reference_dataset.num_frames - 1) // vae_temporal + 1
            cp_frame_div = self.pipe.cp_frame_divisor(num_latent_frames)
            if self.pipe.transformer.frames_per_block is not None:
                # Block-causal CP: cp_size must also divide num_blocks so that
                # each rank gets a whole number of temporal blocks.
                num_blocks = num_latent_frames // self.pipe.transformer.frames_per_block
                cp_frame_div = math.gcd(cp_frame_div, num_blocks)
            cp_group_size = self._choose_cp_group_size(
                cp_frame_div,
                world_size,
                len(task_specs),
            )
            num_groups = world_size // cp_group_size
            group_index = rank // cp_group_size
            group_rank = rank % cp_group_size
            # Ranks that don't fit into a complete group are spectators.
            # Spectators run a tiny dummy forward to keep FSDP all-gathers in sync
            # without bottlenecking CP-active ranks (FSDP collectives are
            # input-size-independent).
            is_spectator = group_index >= num_groups
            num_spectators = world_size - num_groups * cp_group_size
            max_validation_task_groups = getattr(self.args, "max_validation_task_groups", None)
            if max_validation_task_groups is None or max_validation_task_groups <= 0:
                active_task_groups = num_groups
            else:
                active_task_groups = min(num_groups, max_validation_task_groups)
            active_task_groups = max(1, active_task_groups)
            rounds = math.ceil(len(task_specs) / active_task_groups)

            if rank == 0:
                print(
                    f"Validation CP: cp_size={cp_group_size}, groups={num_groups}, "
                    f"active_task_groups={active_task_groups}, "
                    f"spectators={num_spectators}, tasks={len(task_specs)}, "
                    f"rounds={rounds}"
                )

            dummy_task = self._build_minimal_validation_task(
                "__dummy__",
                reference_dataset.num_frames,
                num_latent_frames,
                self._validation_prompt_dim(),
                num_inference_steps,
                self.accelerator.device,
                self.pipe.vae.dtype,
            )

            if cp_group_size > 1:
                # All ranks must participate in DeviceMesh creation (it involves
                # dist.new_group which is a world-collective in some PyTorch versions).
                # Build the full list of CP group rank-lists so every rank calls
                # new_group the same number of times.
                all_group_ranks = [list(range(g * cp_group_size, (g + 1) * cp_group_size)) for g in range(num_groups)]
                # Spectators form their own singleton group(s) for the collective.
                spectator_ranks = list(range(num_groups * cp_group_size, world_size))
                if spectator_ranks:
                    all_group_ranks.append(spectator_ranks)

                my_mesh = None
                for group_ranks in all_group_ranks:
                    mesh = DeviceMesh(self.accelerator.device.type, group_ranks)
                    if rank in group_ranks:
                        my_mesh = mesh

                if not is_spectator:
                    self.pipe.transformer.enable_context_parallel(my_mesh)

            try:
                # Round-robin tasks across active groups; inactive groups run a
                # minimal forward so FSDP collectives stay aligned without
                # increasing full-resolution validation memory fanout.
                # The sync group uses a longer timeout because only group_rank 0 saves results.
                for round_idx in range(rounds):
                    barrier_if_distributed(validation_sync_group)
                    task_index = round_idx * active_task_groups + (group_index if not is_spectator else 0)
                    is_real_task = (
                        not is_spectator and group_index < active_task_groups and task_index < len(task_specs)
                    )

                    if is_real_task:
                        task = self._build_validation_task(
                            val_datasets,
                            task_specs[task_index],
                            validation_index,
                            num_inference_steps,
                        )
                    else:
                        task = dummy_task

                    output = self.pipe.forward_inference(
                        rendered_rgb=task.rendered_rgb,
                        rendered_opacity=task.rendered_opacity,
                        neighbors=task.neighbors,
                        camera_rays=task.camera_rays,
                        w2cs=task.w2cs,
                        neighbor_w2cs=task.neighbor_w2cs,
                        Ks=task.Ks,
                        neighbor_Ks=task.neighbor_Ks,
                        prompt=task.prompt,
                        num_inference_steps=task.num_inference_steps,
                        show_progress=rank == 0,
                        progress_bar_leave=False,
                    )

                    if is_real_task and group_rank == 0:
                        self._save_validation_result(output, task, save_dir)

                    del output
                    del task
                    torch.cuda.empty_cache()
                    gc.collect()
            finally:
                if cp_group_size > 1 and not is_spectator:
                    self.pipe.transformer.disable_context_parallel()

            barrier_if_distributed(validation_sync_group)
        finally:
            if validation_sync_group is not None:
                dist.destroy_process_group(validation_sync_group)

    def _save_validation_result(self, output: torch.Tensor, task: ValidationTask, save_dir: Path) -> None:
        result_rows = [
            torch.cat(
                [output.cpu().squeeze(0), task.rgb_gt.cpu().squeeze(0), task.rendered_rgb.cpu().squeeze(0)], dim=-1
            )
        ]
        if task.neighbors is not None:
            for neighbor_index in range(0, task.neighbors.shape[1], 3):
                result_rows.append(
                    torch.cat([x.cpu() for x in task.neighbors[0, neighbor_index : neighbor_index + 3]], dim=-1)
                    .unsqueeze(0)
                    .expand(result_rows[0].shape[0], -1, -1, -1)
                )
        result = torch.cat(result_rows, dim=-2)
        if result.shape[0] == 1:
            Image.fromarray((result.clamp(0, 1).squeeze(0).permute(1, 2, 0) * 255).byte().numpy()).save(
                save_dir / f"{task.key}.png"
            )
        else:
            save_video(result.clamp(0, 1), save_dir / f"{task.key}.mp4", fps=15)

    def train(self) -> None:
        pbar = tqdm(total=self.args.max_iterations, disable=not self.accelerator.is_local_main_process)
        pbar.update(self.get_step())
        while self.get_step() < self.args.max_iterations:
            data_loading_start_time = time.time()
            for data in self.train_dataloader:
                log_dict = {"train/data_time": time.time() - data_loading_start_time}
                if self.get_step() % 20 == 0:
                    torch.cuda.empty_cache()

                with self.accelerator.accumulate(*self.to_accumulate()):
                    loss, metrics = self.training_batch(data)
                    log_dict.update({f"train/{k}": v for k, v in metrics.items()})
                    if not torch.isfinite(loss).all():
                        torch.save(
                            [data, loss],
                            self.args.project_dir
                            / f"train_error-{self.accelerator.process_index}-{self.get_step()}.pt",
                        )
                        raise ValueError(
                            f"{self.accelerator.process_index} {self.get_step()}: Loss is not finite: {loss}"
                        )

                    log_dict["train/loss"] = loss.item()
                    self.accelerator.backward(loss)

                    self.on_before_optimizer_step()

                    if self.args.check_state_finite:
                        for i, model in enumerate(self.accelerator._models):
                            for name, param in model.named_parameters():
                                if param.grad is not None and (not torch.isfinite(param.grad).all()):
                                    torch.save(
                                        [data, loss, param],
                                        self.args.project_dir
                                        / f"train_error-grad-{i}-{self.accelerator.process_index}-{self.get_step()}.pt",
                                    )
                                    raise ValueError(f"Model {i} parameter gradient {name} contains non-finite values")

                    for optimizer in self.optimizers:
                        optimizer.step()
                    for scheduler in self.schedulers:
                        scheduler.step()

                    if self.args.check_state_finite:
                        for i, model in enumerate(self.accelerator._models):
                            for name, param in model.named_parameters():
                                if not torch.isfinite(param).all():
                                    torch.save(
                                        [data, loss, param],
                                        self.args.project_dir
                                        / f"train_error-param-{i}-{self.accelerator.process_index}-{self.get_step()}.pt",
                                    )
                                    raise ValueError(f"Model {i} parameter {name} contains non-finite values")

                    self.on_before_zero_grad()

                    for optimizer in self.optimizers:
                        optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    # Check only at synchronized step boundaries so all ranks
                    # save the same logical iteration before Slurm requeues.
                    if self._should_checkpoint_for_timeout():
                        self.accelerator.print(
                            f"Saving checkpoint at iteration {self.get_step()} and exiting with timeout signal for requeue"
                        )
                        self._save_state_with_commit_marker(self.get_step())
                        self.accelerator.print(
                            f"Exiting with timeout signal for requeue at iteration {self.get_step()}"
                        )
                        sys.exit(124)

                    if self.get_step() % self.args.save_steps == 0:
                        self.accelerator.project_configuration.iteration = self.get_step()
                        torch.cuda.empty_cache()
                        gc.collect()
                        self._save_state_with_commit_marker(self.get_step())

                    if self.get_step() % self.args.validation_steps == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                        # Validation step 100 uses logical sample 0, step 200
                        # uses sample 1, etc. This preserves the old iterator
                        # ordering without materializing all batches on all ranks.
                        validation_index = max(self.get_step() // self.args.validation_steps - 1, 0)
                        save_dir = self.args.project_dir / "val" / f"step_{self.get_step():06d}"

                        self.validation_batches(self.val_datasets, validation_index, save_dir)

                        if self.accelerator.is_main_process and uses_wandb_logging(self.args.log_with):
                            first_val = self._load_validation_batch(
                                self.val_datasets,
                                min(self.val_datasets.keys()),
                                validation_index,
                            )
                            caption = f"{first_val['prompt_file'][0]}: {first_val['prompt'][0]}"
                            for file in save_dir.glob("*.png"):
                                log_dict[f"val/{file.stem}"] = wandb.Image(file, caption=caption)
                            for file in save_dir.glob("*.mp4"):
                                log_dict[f"val/{file.stem}"] = wandb.Video(
                                    str(file), caption=caption, fps=15, format="mp4"
                                )
                            del first_val

                        torch.cuda.empty_cache()
                        gc.collect()

                    if tracker_logging_enabled(self.args.log_with):
                        self.accelerator.log(log_dict, step=self.get_step())
                    # Barrier after logging: rank 0 may spend extra time uploading
                    # validation images/videos via wandb while other ranks only log
                    # scalars. Without this barrier the faster ranks re-enter the
                    # dataloader (triggering accelerate's RNG-sync broadcast) before
                    # rank 0 is ready, causing an NCCL timeout.
                    barrier_if_distributed()
                    pbar.update(1)
                    data_loading_start_time = time.time()

        self._save_state_with_commit_marker(self.get_step())

    def get_step(self) -> int:
        return self.step_offset + self.accelerator.step // self.args.gradient_accumulation_steps
