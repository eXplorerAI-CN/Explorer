#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run ArtiFixer3D distillation and prepare ArtiFixer3D+ inference metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_processing import artifixer3d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene_root",
        type=Path,
        required=True,
        help="Prepared scene root produced by data_processing.prepare_colmap_artifixer_inputs.",
    )
    parser.add_argument(
        "--artifixer_frames_dir",
        type=Path,
        default=None,
        help="ArtiFixer prediction frame directory, usually <run>/<scene>/frames/batch_0000/pred. "
        "Required when --phases includes distill.",
    )
    parser.add_argument(
        "--split_path",
        type=Path,
        default=None,
        help="Prepared split JSON. Defaults to <scene_root>/split.json.",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Scene id to process. Required only when the split contains multiple scenes.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="ArtiFixer3D output root. Defaults to <scene_root>/artifixer3d.",
    )
    parser.add_argument(
        "--artifixer3d_plus_inference_split_path",
        type=Path,
        default=None,
        help="Output reconstructed_colmap split for ArtiFixer3D+ inference. "
        "Defaults to <scene_root>/split_artifixer3d_plus.json.",
    )
    parser.add_argument(
        "--render_trajectory_path",
        type=Path,
        default=None,
        help=(
            "Optional transforms-style trajectory for post-training ArtiFixer3D rendering. "
            "Defaults to the prepared distillation trajectory from the split."
        ),
    )
    parser.add_argument(
        "--base_checkpoint",
        type=Path,
        default=None,
        help="Optional initial 3DGRUT checkpoint to resume from. Defaults to training ArtiFixer3D from scratch.",
    )
    parser.add_argument(
        "--artifixer3d_steps",
        type=int,
        default=30000,
        help="3DGRUT training iterations for the ArtiFixer3D checkpoint.",
    )
    parser.add_argument(
        "--config_name",
        default="apps/colmap_3dgut_sparse_mcmc_lpips",
        help="3DGRUT config used for ArtiFixer3D pseudo-supervised distillation.",
    )
    parser.add_argument(
        "--phases",
        default="distill,render,prepare_artifixer3d_plus",
        help="Comma-separated phases to run. Valid phases: distill, render, prepare_artifixer3d_plus.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        default=False,
        help="Regenerate ArtiFixer3D outputs even when existing outputs are present.",
    )
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases logging for the 3DGRUT distillation run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifixer3d.run_artifixer3d(args)


if __name__ == "__main__":
    main()
