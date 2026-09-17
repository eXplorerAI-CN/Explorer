# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import zipfile


def _join_zip_member(root: str, *parts: str) -> str:
    return "/".join(part for part in (root, *parts) if part)


def load_scene_transforms_from_zip(zip_file: zipfile.ZipFile, scene_name: str) -> tuple[dict, str]:
    names = set(zip_file.namelist())
    candidates = [
        (f"{scene_name}/nerfstudio/transforms.json", scene_name),
        (f"{scene_name}/transforms.json", scene_name),
        ("nerfstudio/transforms.json", ""),
        ("transforms.json", ""),
    ]
    for transforms_member, scene_root in candidates:
        if transforms_member in names:
            with zip_file.open(transforms_member, "r") as f:
                return json.load(f), scene_root

    formatted = "\n  ".join(transforms_member for transforms_member, _ in candidates)
    raise FileNotFoundError(f"transforms.json not found in {zip_file.filename}. Tried:\n  {formatted}")


def scene_zip_member(scene_root: str, *parts: str) -> str:
    return _join_zip_member(scene_root, *parts)


def downsampled_image_path(file_path: str, downsample_factor: int | float = 4) -> str:
    normalized = file_path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if float(downsample_factor) == 1.0:
        return normalized

    factor = int(downsample_factor)
    if float(factor) != float(downsample_factor):
        raise ValueError(f"downsample_factor must be an integer or 1.0, got {downsample_factor}")
    target = f"images_{factor}"
    parts = normalized.split("/")
    for index, part in enumerate(parts):
        if part == target:
            return normalized
        if part == "images":
            parts[index] = target
            return "/".join(parts)
    raise ValueError(f"Expected an images/ component in frame path: {file_path}")
