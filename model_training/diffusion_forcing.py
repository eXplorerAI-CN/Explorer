# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import torch

from model_training.trainers.trainer import Trainer
from model_training.utils.train_utils import (
    ResumeState,
    barrier_if_distributed,
    get_accelerator,
    get_common_opts,
    get_pipe,
    get_run_id_and_should_resume,
    get_train_dataloader,
    get_val_datasets,
    load_model_weights_from_dcp,
    maybe_write_run_id,
    resume_training_from_checkpoint,
)


def main(args: argparse.Namespace):
    run_id, should_resume = get_run_id_and_should_resume(args)
    accelerator = get_accelerator(args, run_id)
    pipe = get_pipe(args, False, args.frames_per_block, accelerator.device)
    latent_num_frames = (args.num_frames - 1) // pipe.vae.config.scale_factor_temporal + 1
    assert (
        latent_num_frames % args.frames_per_block == 0
    ), f"Number of frames ({args.num_frames}) must be divisible by latent frames per block ({args.frames_per_block})"

    optimizer = torch.optim.AdamW(
        pipe.transformer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    train_dataloader = get_train_dataloader(args, accelerator, args.frames_per_block)

    # Keep ranks aligned after model and dataloader setup; large model loading can
    # otherwise leave faster ranks waiting inside distributed preparation.
    barrier_if_distributed()

    transformer, optimizer, train_dataloader, scheduler = accelerator.prepare(
        pipe.transformer, optimizer, train_dataloader, scheduler
    )

    # Should only load state after prepare is called
    resume_state = ResumeState()
    if should_resume:
        train_dataloader, resume_state = resume_training_from_checkpoint(args, accelerator, train_dataloader)
    else:
        load_model_weights_from_dcp(transformer, args.base_checkpoint_dir)

    pipe.transformer = transformer

    maybe_write_run_id(accelerator, args.project_dir, run_id, args.log_with)

    # Build validation datasets after loading the checkpoint so validation uses
    # the final split/model configuration.
    val_datasets = get_val_datasets(args, accelerator, args.num_frames, None)

    trainer = Trainer(
        args,
        accelerator,
        optimizer,
        scheduler,
        train_dataloader,
        val_datasets,
        pipe,
        args.frames_per_block,
        resume_state.step_offset,
    )
    trainer.train()


if __name__ == "__main__":
    parser = get_common_opts()

    parser.add_argument("--project_dir", required=True, type=Path)
    parser.add_argument("--base_checkpoint_dir", required=True, type=Path)
    parser.add_argument("--frames_per_block", default=7, type=int)

    parser.add_argument("--max_iterations", default=10000, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--validation_steps", default=1000, type=int)

    main(parser.parse_args())
