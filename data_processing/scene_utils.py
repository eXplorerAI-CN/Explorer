# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from model_training.data.scene_utils import downsampled_image_path, load_scene_transforms_from_zip, scene_zip_member


def read_scene_list(path: Path) -> list[str]:
    scenes: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            scenes.append(line)
    return scenes


def discover_scene_zips(dl3dv_dir: Path) -> list[Path]:
    return sorted(path for path in dl3dv_dir.rglob("*.zip") if path.is_file())


def resolve_scene_path(dl3dv_dir: Path, scene: str) -> Path:
    scene_path = Path(scene)
    candidates: list[Path] = []
    if scene_path.is_absolute():
        candidates.append(scene_path)
    else:
        candidates.append(dl3dv_dir / scene_path)
        if scene_path.suffix == ".zip" and len(scene_path.parts) == 1:
            candidates.extend(sorted(dl3dv_dir.glob(f"*/{scene_path.name}")))
        elif scene_path.suffix != ".zip":
            candidates.append(dl3dv_dir / scene_path.with_suffix(".zip"))
            candidates.extend(sorted(dl3dv_dir.glob(f"*/{scene}.zip")))
            candidates.extend(sorted(dl3dv_dir.glob(f"*/{scene}")))

    matches = []
    seen = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        formatted = "\n  ".join(str(path) for path in matches)
        raise ValueError(f"Scene {scene!r} is ambiguous under {dl3dv_dir}; use a relative path:\n  {formatted}")
    raise FileNotFoundError(f"Could not resolve scene {scene!r} under {dl3dv_dir}")


def scene_name(scene_path: Path) -> str:
    return scene_path.stem if scene_path.suffix == ".zip" else scene_path.name


def scene_relative_parent(dl3dv_dir: Path, scene_path: Path) -> Path:
    try:
        return scene_path.parent.relative_to(dl3dv_dir)
    except ValueError:
        return Path()


def scene_dl3dv_subdir(dl3dv_dir: Path, scene_path: Path, default: str = "scenes") -> str:
    rel_parent = scene_relative_parent(dl3dv_dir, scene_path)
    return rel_parent.parts[0] if rel_parent.parts else default


def load_scene_transforms_from_dir(scene_dir: Path) -> dict:
    candidates = [
        scene_dir / "nerfstudio" / "transforms.json",
        scene_dir / "transforms.json",
    ]
    for transforms_path in candidates:
        if transforms_path.is_file():
            with transforms_path.open() as f:
                return json.load(f)
    formatted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"transforms.json not found. Tried:\n  {formatted}")


def load_source_scene_transforms(dl3dv_dir: Path, source_subdir: str, scene_name: str) -> tuple[dict, str]:
    """Load transforms for a scene under a DL3DV root and return (transforms, zip_member_prefix)."""
    zip_path = dl3dv_dir / source_subdir / f"{scene_name}.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"source zip missing: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        transforms, scene_root = load_scene_transforms_from_zip(zip_file, scene_name)
    prefix = f"{scene_root}/" if scene_root else ""
    return transforms, prefix


def safe_extractall(zip_file: zipfile.ZipFile, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    for member in zip_file.infolist():
        target = (output_dir / member.filename).resolve()
        if target != output_root and output_root not in target.parents:
            raise ValueError(f"Refusing to extract zip member outside {output_dir}: {member.filename}")
    zip_file.extractall(output_dir)


def extract_zip(zip_path: Path, output_dir: Path, *, skip_existing_scene_dir: bool = True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = output_dir / zip_path.stem
    if skip_existing_scene_dir and scene_dir.exists():
        return scene_dir

    with zipfile.ZipFile(zip_path, "r") as zf:
        safe_extractall(zf, output_dir)

    if scene_dir.exists():
        return scene_dir
    return output_dir
