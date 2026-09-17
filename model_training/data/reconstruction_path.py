# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def parse_reconstruction_hdf5_path(hdf5_path: str | Path) -> tuple[str, str, int]:
    """Parse current data_<scene_id>_<half>_<num_views>.h5 reconstruction names."""
    path = Path(hdf5_path)
    name = path.name
    if not name.endswith(".h5"):
        raise ValueError(f"Expected an .h5 reconstruction file, got {path}")

    stem = name[: -len(".h5")]

    try:
        prefix_scene, scene_half, num_selected_indices = stem.rsplit("_", 2)
    except ValueError as e:
        raise ValueError(f"Unexpected reconstruction filename format: {path}") from e

    if not prefix_scene.startswith("data_"):
        raise ValueError(f"Unexpected reconstruction filename format: {path}")

    scene_id = prefix_scene[len("data_") :]
    if not scene_id:
        raise ValueError(f"Unexpected reconstruction filename format: {path}")

    try:
        num_views = int(num_selected_indices)
    except ValueError as e:
        raise ValueError(f"Unexpected reconstruction filename format: {path}") from e
    return scene_id, scene_half, num_views
