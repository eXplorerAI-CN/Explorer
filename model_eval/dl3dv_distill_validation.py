# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from pathlib import Path


CAMERA_KEYS = (
    "camera_angle_x",
    "camera_angle_y",
    "fl_x",
    "fl_y",
    "cx",
    "cy",
    "w",
    "h",
    "k1",
    "k2",
    "k3",
    "k4",
    "p1",
    "p2",
)


def _load_transforms(scene_dataset_path: Path) -> dict:
    transforms_path = scene_dataset_path / "transforms.json"
    with transforms_path.open() as f:
        transforms = json.load(f)

    frames = transforms.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {transforms_path}")
    return transforms


def _load_frame_paths(scene_dataset_path: Path) -> list[str]:
    transforms = _load_transforms(scene_dataset_path)
    frames = transforms.get("frames", [])
    return [frame["file_path"] for frame in frames]


def _load_frame_count(scene_dataset_path: Path) -> int:
    transforms = _load_transforms(scene_dataset_path)
    frames = transforms.get("frames", [])
    return len(frames)


def _frame_signature(transforms: dict, frame: dict) -> dict:
    signature = {"file_path": frame.get("file_path"), "transform_matrix": frame.get("transform_matrix")}
    for key in CAMERA_KEYS:
        if key in frame:
            signature[key] = frame[key]
        elif key in transforms:
            signature[key] = transforms[key]
    return signature


def _override_frame_indices(scene_image_override: Path) -> list[int]:
    indices = []
    for path in scene_image_override.glob("*.png"):
        if path.stem.endswith("_mask") or not path.stem.isdigit():
            continue
        indices.append(int(path.stem))

    if not indices:
        raise ValueError(f"No numeric PNG override frames found in {scene_image_override}")
    return sorted(indices)


def validate_override_frame_indices(
    scene_dataset_path: str | Path,
    scene_image_override: str | Path,
    reference_scene_dataset_path: str | Path | None = None,
) -> None:
    scene_dataset_path = Path(scene_dataset_path)
    scene_image_override = Path(scene_image_override)
    frame_count = _load_frame_count(scene_dataset_path)
    indices = _override_frame_indices(scene_image_override)
    invalid_indices = [idx for idx in indices if idx >= frame_count]
    if invalid_indices:
        max_valid = frame_count - 1
        preview = ", ".join(str(idx) for idx in invalid_indices[:10])
        raise ValueError(
            "Image override directory contains frame indices outside the scene frame table: "
            f"{preview}. Scene has {frame_count} frames, so the max valid index is {max_valid}. "
            "This usually means DATASET_PATH does not match the ArtiFixer render frame table."
        )

    if reference_scene_dataset_path is None:
        return

    transforms = _load_transforms(scene_dataset_path)
    frames = transforms.get("frames", [])
    frame_paths = [frame["file_path"] for frame in frames]
    reference_scene_dataset_path = Path(reference_scene_dataset_path)
    reference_transforms = _load_transforms(reference_scene_dataset_path)
    reference_frames = reference_transforms.get("frames", [])
    reference_frame_paths = [frame["file_path"] for frame in reference_frames]
    invalid_reference_indices = [idx for idx in indices if idx >= len(reference_frame_paths)]
    if invalid_reference_indices:
        preview = ", ".join(str(idx) for idx in invalid_reference_indices[:10])
        raise ValueError(
            "Image override directory contains frame indices outside the reference frame table: "
            f"{preview}. Reference has {len(reference_frame_paths)} frames."
        )

    mismatched_indices = [
        idx
        for idx in indices
        if _frame_signature(transforms, frames[idx]) != _frame_signature(reference_transforms, reference_frames[idx])
    ]
    if mismatched_indices:
        idx = mismatched_indices[0]
        detail = (
            f"{frame_paths[idx]} != {reference_frame_paths[idx]}"
            if frame_paths[idx] != reference_frame_paths[idx]
            else "camera metadata differs"
        )
        raise ValueError(
            "Scene frame table has a different frame at override index "
            f"{idx}: {detail}. "
            "Use a DATASET_PATH with the same transforms.json ordering as the ArtiFixer renders."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate DL3DV distillation override frame indices.")
    parser.add_argument("scene_dataset_path", type=Path)
    parser.add_argument("scene_image_override", type=Path)
    parser.add_argument("--reference_scene_dataset_path", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        validate_override_frame_indices(
            args.scene_dataset_path,
            args.scene_image_override,
            reference_scene_dataset_path=args.reference_scene_dataset_path,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
