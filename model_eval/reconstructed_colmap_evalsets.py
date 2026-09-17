# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from model_eval.datasets.reconstructed_colmap_eval import (
    DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS,
    ReconstructedColmapEvalDataset,
)

RECONSTRUCTED_COLMAP_EVALSETS = ("reconstructed_colmap",)


def is_reconstructed_colmap_evalset(evalset: str) -> bool:
    return evalset in RECONSTRUCTED_COLMAP_EVALSETS


def create_reconstructed_colmap_dataset(args, selection_mode, max_test_frames, include_all_frames, rank: int):
    return ReconstructedColmapEvalDataset(
        split="test",
        split_path=args.split_path,
        num_views=args.num_views,
        neighbor_selection_mode=selection_mode,
        max_test_frames=max_test_frames,
        include_all_frames=include_all_frames,
        use_target_indices=args.render_trajectory == "trajectory",
        filter_scene_id=args.scene_id,
        generator=None,
        verbose=(rank == 0),
    )
