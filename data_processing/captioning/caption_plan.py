# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caption output naming and variant helpers."""

from __future__ import annotations

from pathlib import Path


def caption_output_path(
    scene_output_dir: Path,
    num_frames: int | None,
    frame_stride: int,
    *,
    revisit: bool = False,
    reverse: bool = False,
) -> Path:
    name_parts = [
        f"frames_{num_frames if num_frames is not None and num_frames > 0 else 'all'}",
        f"stride_{frame_stride}",
    ]
    if revisit:
        name_parts.append("revisit")
    if reverse:
        name_parts.append("reverse")
    return scene_output_dir / ("_".join(name_parts) + ".h5")


def caption_variants(max_frame_stride: int, *, include_reverse: bool, include_revisit: bool) -> list[tuple[int, bool, bool]]:
    variants: list[tuple[int, bool, bool]] = []
    reverse_values = [False, True] if include_reverse else [False]
    revisit_values = [False, True] if include_revisit else [False]
    for frame_stride in range(1, max_frame_stride + 1):
        for revisit in revisit_values:
            for reverse in reverse_values:
                variants.append((frame_stride, revisit, reverse))
    return variants
