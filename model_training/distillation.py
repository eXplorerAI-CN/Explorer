# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import torch
from accelerate.utils.fsdp_utils import fsdp2_prepare_model
from diffusers import WanTransformer3DModel

from model_training.net.transformer import ArtifixerTransformer
from model_training.trainers.dmd import DMD
from model_training.utils.ema import DTensorFastEmaModelUpdater
from model_training.utils.train_utils import (
    ResumeState,
    barrier_if_distributed,
    get_accelerator,
    get_common_opts,
    get_kv_cache_pipe,
    get_run_id_and_should_resume,
    get_train_dataloader,
    get_val_datasets,
    load_model_weights_from_dcp,
    maybe_write_run_id,
    resume_training_from_checkpoint,
)


def main(args: argparse.Namespace):
    if args.model_id_critic is None:
        args.model_id_critic = args.model_id

    run_id, should_resume = get_run_id_and_should_resume(args)
    accelerator = get_accelerator(args, run_id)
    pipe = get_kv_cache_pipe(args, accelerator.device)

    latent_num_frames = (args.num_frames - 1) // pipe.vae.config.scale_factor_temporal + 1
    assert (
        latent_num_frames % args.frames_per_block == 0
    ), f"Number of frames ({args.num_frames}) must be divisible by latent frames per block ({args.frames_per_block})"
    transformer_real_score = WanTransformer3DModel.from_config(
        args.model_id_critic, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(accelerator.device)
    real_score = ArtifixerTransformer(
        transformer_real_score,
        frames_per_block=None,
        local_attn_size=None,
        sink_size=None,
        vae_scale_factor_spatial=pipe.vae.config.scale_factor_spatial,
        vae_scale_factor_temporal=pipe.vae.config.scale_factor_temporal,
        gradient_checkpointing=False,
        checkpoint_every_n_blocks=1,
    )
    real_score.eval()
    real_score.requires_grad_(False)

    transformer_fake_score = WanTransformer3DModel.from_config(
        args.model_id_critic, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    fake_score = ArtifixerTransformer(
        transformer_fake_score,
        frames_per_block=None,
        local_attn_size=None,
        sink_size=None,
        vae_scale_factor_spatial=pipe.vae.config.scale_factor_spatial,
        vae_scale_factor_temporal=pipe.vae.config.scale_factor_temporal,
        gradient_checkpointing=args.gradient_checkpointing,
        checkpoint_every_n_blocks=args.checkpoint_every_n_blocks,
    )

    optimizer = torch.optim.AdamW(pipe.transformer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    optimizer_fake_score = torch.optim.AdamW(
        fake_score.parameters(), lr=args.learning_rate_critic, weight_decay=args.weight_decay
    )
    scheduler_fake_score = torch.optim.lr_scheduler.ConstantLR(optimizer_fake_score)

    train_dataloader = get_train_dataloader(args, accelerator, args.frames_per_block)

    # Keep ranks aligned after model and dataloader setup; large model loading can
    # otherwise leave faster ranks waiting inside distributed preparation.
    barrier_if_distributed()

    transformer, optimizer, scheduler, train_dataloader = accelerator.prepare(
        pipe.transformer,
        optimizer,
        scheduler,
        train_dataloader,
    )

    fake_score, optimizer_fake_score, scheduler_fake_score = accelerator.prepare(
        fake_score, optimizer_fake_score, scheduler_fake_score
    )

    if args.ema_weight > 0:
        transformer_ema = WanTransformer3DModel.from_config(
            args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        ema = ArtifixerTransformer(
            transformer_ema,
            frames_per_block=args.frames_per_block,
            local_attn_size=args.local_attn_size,
            sink_size=args.sink_size,
            vae_scale_factor_spatial=pipe.vae.config.scale_factor_spatial,
            vae_scale_factor_temporal=pipe.vae.config.scale_factor_temporal,
            gradient_checkpointing=args.gradient_checkpointing,
            checkpoint_every_n_blocks=args.checkpoint_every_n_blocks,
        )
        ema.requires_grad_(False)
        # The EMA model is not optimized directly, but it still needs the same
        # FSDP2 wrapping as the trainable transformer.
        accelerator._prepare_fsdp2(ema)
        ema_worker = DTensorFastEmaModelUpdater()
    else:
        ema = None
        ema_worker = None

    # Should only load state after prepare is called
    resume_state = ResumeState()
    if should_resume:
        train_dataloader, resume_state = resume_training_from_checkpoint(args, accelerator, train_dataloader)
    else:
        load_model_weights_from_dcp(transformer, args.base_checkpoint_dir)
        load_model_weights_from_dcp(fake_score, args.base_checkpoint_dir_critic)

        if ema_worker is not None:
            ema_worker.copy_to(src_model=transformer, tgt_model=ema)

    pipe.transformer = transformer

    real_score = fsdp2_prepare_model(accelerator, real_score)
    load_model_weights_from_dcp(real_score, args.base_checkpoint_dir_critic)

    maybe_write_run_id(accelerator, args.project_dir, run_id, args.log_with)

    # Build validation datasets after loading the checkpoint so validation uses
    # the final split/model configuration.
    val_datasets = get_val_datasets(args, accelerator, None, args.frames_per_block)

    trainer = DMD(
        args,
        accelerator,
        [optimizer, optimizer_fake_score],
        [scheduler, scheduler_fake_score],
        train_dataloader,
        val_datasets,
        pipe,
        real_score,
        fake_score,
        ema,
        ema_worker,
        resume_state.step_offset,
    )
    trainer.train()


if __name__ == "__main__":
    parser = get_common_opts()

    parser.add_argument("--project_dir", required=True, type=Path)
    parser.add_argument("--base_checkpoint_dir", required=True, type=Path)
    parser.add_argument("--base_checkpoint_dir_critic", required=True, type=Path)

    # Available models: Wan-AI/Wan2.1-T2V-1.3B-Diffusers, Wan-AI/Wan2.1-T2V-14B-Diffusers
    parser.add_argument(
        "--model_id_critic",
        default=None,
        type=str,
        help="Model config for the DMD real/fake score critics. Defaults to --model_id.",
    )

    parser.add_argument("--num_inference_steps", default=4, type=int)
    parser.add_argument("--frames_per_block", default=7, type=int)
    parser.add_argument("--local_attn_size", default=21, type=int)
    parser.add_argument("--sink_size", default=7, type=int)
    parser.add_argument("--text_guidance_scale", default=3.0, type=float)
    parser.add_argument("--timestep_shift", default=5.0, type=float)
    parser.add_argument("--dfake_gen_update_ratio", default=5, type=int)

    parser.add_argument("--max_iterations", default=400, type=int)
    parser.add_argument("--learning_rate", default=2e-6, type=float)
    parser.add_argument("--learning_rate_critic", default=4e-7, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max_grad_norm", default=10.0, type=float)
    parser.add_argument("--ema_weight", default=0.99, type=float)
    parser.add_argument("--ema_start_step", default=200, type=int)

    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--validation_steps", default=100, type=int)

    main(parser.parse_args())
