# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse


DL3DV_RECONSTRUCTION_EVALSETS = ("3dgrut_dl3dv_difix", "artifixer3d_dl3dv_difix", "artifixer3d_dl3dv_ours")

# Evalset naming follows the reconstruction stage and DL3DV split policy:
# - 3dgrut_dl3dv_ours uses the HDF5 outputs from our sparse-reconstruction pipeline.
# - The entries below adapt existing reconstruction-render directories into the reconstructed-COLMAP loader.
RECON_SUBDIR_BY_EVALSET = {
    "3dgrut_dl3dv_difix": None,
    "artifixer3d_dl3dv_difix": "distill_difix_eval",
    "artifixer3d_dl3dv_ours": "distill_ours_eval",
}


def is_dl3dv_reconstruction_evalset(evalset: str) -> bool:
    return evalset in DL3DV_RECONSTRUCTION_EVALSETS


def add_dl3dv_reconstruction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dl3dv_reconstruction_checkpoint",
        type=str,
        default="30000",
        help="Reconstruction output checkpoint iteration to read from DL3DV reconstruction render directories.",
    )
    parser.add_argument(
        "--dl3dv_recon_subdir",
        type=str,
        default=None,
        help="Override the reconstruction experiment directory under each DL3DV scene.",
    )


def create_dl3dv_reconstruction_dataset(args, selection_mode, max_test_frames, include_all_frames, rank: int):
    from model_eval.datasets.dl3dv_reconstruction_eval import DL3DVReconstructionEvalDataset

    recon_subdir = args.dl3dv_recon_subdir or RECON_SUBDIR_BY_EVALSET[args.evalset]
    return DL3DVReconstructionEvalDataset(
        split="test",
        split_path=args.split_path,
        dl3dv_dir=args.dl3dv_dir,
        recon_results_dir=args.recon_results_dir,
        prompt_dir=args.prompt_dir,
        num_views=args.num_views,
        dataset_scaling_factor=args.dataset_scaling_factor,
        checkpoint=args.dl3dv_reconstruction_checkpoint,
        neighbor_selection_mode=selection_mode,
        max_test_frames=max_test_frames,
        include_all_frames=include_all_frames,
        filter_scene_id=args.scene_id,
        generator=None,
        verbose=(rank == 0),
        recon_subdir=recon_subdir,
        use_difix_frame_remap=args.evalset.endswith("_difix"),
    )
