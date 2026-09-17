# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from argparse import Namespace
from pathlib import Path


def add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint_pt",
        type=Path,
        default=None,
        help="Single-file transformer state dict exported from a DCP/FSDP checkpoint.",
    )


def validate_checkpoint_args(parser: argparse.ArgumentParser, args: Namespace) -> None:
    has_checkpoint_dir = args.checkpoint_dir is not None
    has_checkpoint_pt = args.checkpoint_pt is not None
    if has_checkpoint_dir == has_checkpoint_pt:
        parser.error("exactly one of --checkpoint_dir or --checkpoint_pt is required")


def checkpoint_output_name(args: Namespace) -> str:
    if args.checkpoint_pt is not None:
        return args.checkpoint_pt.stem

    ckpt_parent = args.checkpoint_dir.parent.name
    ckpt_name = args.checkpoint_dir.name
    return f"{ckpt_parent}_{ckpt_name}" if ckpt_parent.startswith("checkpoint") else ckpt_name


def _torch_load_state_dict(checkpoint_pt: Path):
    import torch

    try:
        return torch.load(checkpoint_pt, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(checkpoint_pt, map_location="cpu", weights_only=True)


def load_model_weights_from_pt(model, checkpoint_pt: Path | str) -> None:
    state_dict = _torch_load_state_dict(Path(checkpoint_pt))
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state dict in {checkpoint_pt}, got {type(state_dict).__name__}")
    model.load_state_dict(state_dict)


def load_transformer_checkpoint(
    model,
    args: Namespace,
    *,
    dcp_loader=None,
    pt_loader=None,
) -> None:
    if args.checkpoint_pt is not None:
        loader = pt_loader if pt_loader is not None else load_model_weights_from_pt
        loader(model, args.checkpoint_pt)
        return

    if dcp_loader is None:
        from model_training.utils.train_utils import load_model_weights_from_dcp

        dcp_loader = load_model_weights_from_dcp
    dcp_loader(model, args.checkpoint_dir)
