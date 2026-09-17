# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared NerfBusters dataset conventions."""

from __future__ import annotations

import os

NERFBUSTERS_HIGH_RES_SCENES = frozenset({"aloe", "car", "garbage", "table"})

NERFBUSTERS_SCENES = (
    "aloe",
    "art",
    "car",
    "century",
    "flowers",
    "garbage",
    "picnic",
    "pikachu",
    "pipe",
    "plant",
    "roses",
    "table",
)


def nerfbusters_downsample_factor(scene_id: str) -> int:
    """Return the 3DGRUT image downsample factor for paper-style NerfBusters eval."""
    return 2 if scene_id in NERFBUSTERS_HIGH_RES_SCENES else 1


def nerfbusters_image_folder(scene_id: str) -> str:
    """Return the GT image folder matching the reconstruction downsample factor."""
    downsample_factor = nerfbusters_downsample_factor(scene_id)
    return "images" if downsample_factor == 1 else f"images_{downsample_factor}"


def nerfbusters_gt_relpath(scene_id: str, file_path: str) -> str:
    """Map a transforms.json image path to the paper-style eval image folder."""
    image_folder = nerfbusters_image_folder(scene_id)
    normalized = file_path.lstrip("./")
    if normalized.startswith("images/"):
        return normalized.replace("images/", f"{image_folder}/", 1)
    if normalized.startswith("images_"):
        basename = os.path.basename(normalized)
        return f"{image_folder}/{basename}"
    return normalized
