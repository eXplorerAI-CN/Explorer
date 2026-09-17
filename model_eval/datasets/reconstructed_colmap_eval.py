# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation dataset for COLMAP-scene reconstruction renders.

The prepared split file stores the exact paths for each scene:
    transforms_path
    image_root
    render_dir
    opacity_dir
    selected_indices_path
    target_indices_path (optional; explicit frames to correct)
    prompt_path
    camera_scale
    has_gt (optional; defaults to true)
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image

from model_training.data.scene_utils import scene_zip_member
from model_training.data.utils import (
    InferencePair,
    NeighborSelectionMode,
    compute_camera_rays,
    generate_inference_pairs,
    load_encoded_prompt,
    load_indexed_frames,
    resize_to_multiple_of_16,
    visualize_inference_pairs,
)

DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS = 12


@dataclass(frozen=True)
class ReconstructedColmapScene:
    scene_id: str
    transforms_path: Path
    image_root: Path
    render_dir: Path
    opacity_dir: Path
    selected_indices_path: Path
    target_indices_path: Path | None
    prompt_paths: list[Path]
    camera_scale: float
    has_gt: bool


def resolve_prepared_path(split_root: Path, value: str, field_name: str) -> Path:
    assert isinstance(value, str), f"reconstructed-COLMAP split field {field_name!r} must be a path string"
    path = Path(value)
    return path if path.is_absolute() else split_root / path


def resolve_required_prepared_path(split_root: Path, metadata: dict[str, Any], field_name: str) -> Path:
    assert field_name in metadata, f"Missing required reconstructed-COLMAP split field: {field_name}"
    return resolve_prepared_path(split_root, metadata[field_name], field_name)


def resolve_required_number(metadata: dict[str, Any], field_name: str) -> float:
    assert field_name in metadata, f"Missing required reconstructed-COLMAP split field: {field_name}"
    value = metadata[field_name]
    assert isinstance(value, int | float) and not isinstance(
        value, bool
    ), f"reconstructed-COLMAP split field {field_name!r} must be numeric, got {value!r}"
    return float(value)


def load_frames_from_prepared_paths(image_root: Path, file_paths: list[str]) -> torch.Tensor:
    frames = []
    if image_root.suffix == ".zip":
        scene_root = image_root.stem
        with ZipFile(image_root, "r") as zip_file:
            names = set(zip_file.namelist())
            for file_path in file_paths:
                candidates = (scene_zip_member(scene_root, file_path), scene_zip_member("", file_path))
                matches = [candidate for candidate in candidates if candidate in names]
                assert matches, f"{image_root} is missing frame {file_path!r}"
                with zip_file.open(matches[0], "r") as image_file, Image.open(image_file) as image:
                    frames.append(np.asarray(image.convert("RGB")))
    else:
        for file_path in file_paths:
            path = Path(file_path)
            if not path.is_absolute():
                path = image_root / path
            with Image.open(path) as image:
                frames.append(np.asarray(image.convert("RGB")))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.FloatTensor(np.stack(frames)).permute(0, 3, 1, 2) / 255.0


class ReconstructedColmapEvalDataset(torch.utils.data.Dataset):
    """
    Dataset for evaluating ArtiFixer on COLMAP-scene reconstruction renders.

    Reads from reconstruction outputs:
    - renders/                - Rendered RGB images for all frames
    - opacity/                - Rendered opacity masks
    - selected_indices.json   - Training frame indices (test = complement)
    """

    def __init__(
        self,
        split: str,
        split_path: Path,
        num_views: int | None,
        neighbor_selection_mode: NeighborSelectionMode = NeighborSelectionMode.COVISIBILITY,
        max_test_frames: int | None = None,
        include_all_frames: bool = False,
        use_target_indices: bool = False,
        filter_scene_id: str | None = None,
        generator: torch.Generator | None = None,
        verbose: bool = False,
    ):
        self.generator = generator
        self.num_views = num_views
        self.neighbor_selection_mode = neighbor_selection_mode
        self.include_all_frames = include_all_frames
        self.use_target_indices = use_target_indices

        self._load_scene_data(split_path, split, filter_scene_id, verbose)
        self._generate_inference_items(num_views, max_test_frames, verbose)

    def _load_scene_data(
        self,
        split_path: Path,
        split: str,
        filter_scene_id: str | None,
        verbose: bool,
    ) -> None:
        """Load scene metadata, transforms, and train/test splits."""
        split_path = split_path.resolve()
        split_root = split_path.parent
        with open(split_path) as f:
            split_data = json.load(f)[split]

        self._reset_scene_data()

        assert isinstance(
            split_data, dict
        ), "reconstructed_colmap split must be a mapping from scene id to prepared-scene metadata"

        for scene_id, metadata in split_data.items():
            if filter_scene_id is not None and scene_id != filter_scene_id:
                continue

            scene = self._load_scene_metadata(split_root, scene_id, metadata)
            assert not (
                self.include_all_frames and scene.target_indices_path is not None and not self.use_target_indices
            ), f"Scene {scene_id!r} is a trajectory split; use --render_trajectory=trajectory, not all_frames"
            required_paths = (
                scene.transforms_path,
                scene.image_root,
                scene.render_dir,
                scene.opacity_dir,
                scene.selected_indices_path,
            )
            if scene.target_indices_path is not None:
                required_paths = (*required_paths, scene.target_indices_path)
            for path in (*required_paths, *scene.prompt_paths):
                if not path.exists():
                    raise FileNotFoundError(f"Prepared scene {scene_id!r} references missing path: {path}")

            with open(scene.transforms_path) as f:
                transforms = json.load(f)
            self._register_scene(scene, transforms, verbose)

        if verbose:
            print(f"Found {len(self.scene_ids)} scenes with reconstruction renders")

    @staticmethod
    def _load_scene_metadata(split_root: Path, scene_id: str, metadata: dict[str, Any]) -> ReconstructedColmapScene:
        assert metadata, (
            f"Scene {scene_id!r} is missing prepared metadata. "
            "Re-run data_processing.prepare_colmap_artifixer_inputs with the current code."
        )
        prompt_path = resolve_required_prepared_path(split_root, metadata, "prompt_path")
        has_gt = metadata.get("has_gt", True)
        assert isinstance(has_gt, bool), f"Scene {scene_id!r} has non-boolean has_gt metadata: {has_gt!r}"
        return ReconstructedColmapScene(
            scene_id=scene_id,
            transforms_path=resolve_required_prepared_path(split_root, metadata, "transforms_path"),
            image_root=resolve_required_prepared_path(split_root, metadata, "image_root"),
            render_dir=resolve_required_prepared_path(split_root, metadata, "render_dir"),
            opacity_dir=resolve_required_prepared_path(split_root, metadata, "opacity_dir"),
            selected_indices_path=resolve_required_prepared_path(split_root, metadata, "selected_indices_path"),
            target_indices_path=(
                resolve_prepared_path(split_root, metadata["target_indices_path"], "target_indices_path")
                if "target_indices_path" in metadata
                else None
            ),
            prompt_paths=[prompt_path],
            camera_scale=resolve_required_number(metadata, "camera_scale"),
            has_gt=has_gt,
        )

    def _reset_scene_data(self) -> None:
        self.scene_ids = []
        self.scenes_by_scene_id: dict[str, ReconstructedColmapScene] = {}
        self.transforms_by_scene_id = {}
        self.train_ids_by_scene_id = {}
        self.target_ids_by_scene_id = {}

    def _register_scene(self, scene: ReconstructedColmapScene, transforms: dict[str, Any], verbose: bool) -> None:
        scene_id = scene.scene_id
        self.transforms_by_scene_id[scene_id] = transforms
        train_ids = self._load_train_ids(scene_id, scene.selected_indices_path, verbose)
        target_ids = (
            self._load_target_ids(scene_id, scene.target_indices_path, verbose) if self.use_target_indices else None
        )
        assert (
            not self.use_target_indices or target_ids is not None
        ), f"Scene {scene_id!r} is missing target_indices_path for --render_trajectory=trajectory"
        self.scenes_by_scene_id[scene_id] = scene
        self.train_ids_by_scene_id[scene_id] = train_ids
        self.target_ids_by_scene_id[scene_id] = target_ids
        self.scene_ids.append(scene_id)

    @staticmethod
    def _load_index_set(path: Path) -> set[int]:
        with open(path) as f:
            indices = json.load(f)
        assert isinstance(indices, list), f"{path} must contain a JSON list"
        assert all(
            isinstance(index, int) and not isinstance(index, bool) for index in indices
        ), f"{path} must contain integer frame indices"
        assert len(set(indices)) == len(indices), f"{path} must not contain duplicate frame indices: {indices}"
        return set(indices)

    def _load_train_ids(self, scene_id: str, selected_indices_path: Path, verbose: bool) -> set[int]:
        """Load training frame IDs from selected_indices.json."""
        train_ids = self._load_index_set(selected_indices_path)
        total_frames = len(self.transforms_by_scene_id[scene_id]["frames"])
        assert train_ids, f"Scene {scene_id} selected_indices_path is empty"
        assert (
            min(train_ids) >= 0 and max(train_ids) < total_frames
        ), f"Scene {scene_id} selected indices must be in [0, {total_frames - 1}], got {sorted(train_ids)}"
        test_count = total_frames - len(train_ids)

        if verbose:
            print(f"Scene {scene_id}: {len(train_ids)} train, {test_count} test")

        return train_ids

    def _load_target_ids(self, scene_id: str, target_indices_path: Path | None, verbose: bool) -> set[int] | None:
        if target_indices_path is None:
            return None

        target_ids = self._load_index_set(target_indices_path)
        total_frames = len(self.transforms_by_scene_id[scene_id]["frames"])
        assert target_ids, f"Scene {scene_id} target_indices_path is empty"
        assert (
            min(target_ids) >= 0 and max(target_ids) < total_frames
        ), f"Scene {scene_id} target indices must be in [0, {total_frames - 1}], got {sorted(target_ids)}"
        if verbose:
            print(f"Scene {scene_id}: {len(target_ids)} trajectory target frames")
        return target_ids

    def _generate_inference_items(
        self,
        num_views: int | None,
        max_test_frames: int | None,
        verbose: bool,
    ) -> None:
        """Generate inference pairs for all scenes."""
        self.inference_items: list[tuple[str, InferencePair]] = []

        for scene_id in self.scene_ids:
            transforms = self.transforms_by_scene_id[scene_id]
            train_ids = self.train_ids_by_scene_id[scene_id]
            target_ids = self.target_ids_by_scene_id[scene_id]
            if target_ids is not None:
                assert not (train_ids & target_ids), f"Scene {scene_id} train and target indices overlap"
            total_frames = len(transforms["frames"])
            scene_num_views = self._resolve_num_views(num_views, train_ids, scene_id)

            extrinsics_c2w = None
            if self.neighbor_selection_mode == NeighborSelectionMode.COVISIBILITY:
                extrinsics_c2w = np.array([transforms["frames"][i]["transform_matrix"] for i in range(total_frames)])

            pairs = generate_inference_pairs(
                train_ids=train_ids,
                total_frames=total_frames,
                num_train_context=scene_num_views,
                max_test_frames=max_test_frames,
                selection_mode=self.neighbor_selection_mode,
                extrinsics_c2w=extrinsics_c2w,
                test_ids=target_ids,
                include_all_frames=self.include_all_frames and target_ids is None,
            )

            for pair in pairs:
                pair.scene_id = scene_id
                self.inference_items.append((scene_id, pair))

        if verbose:
            print(f"Generated {len(self.inference_items)} inference items")

    @staticmethod
    def _resolve_num_views(num_views: int | None, train_ids: set[int], scene_id: str) -> int:
        if num_views is None:
            resolved = min(DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS, len(train_ids))
        else:
            resolved = num_views

        assert resolved > 0, f"Scene {scene_id} has no selected views"
        assert resolved <= len(
            train_ids
        ), f"Scene {scene_id} requested {resolved} context views but only {len(train_ids)} are selected"
        return resolved

    def __len__(self) -> int:
        return len(self.inference_items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        scene_id, pair = self.inference_items[idx]

        transforms = self.transforms_by_scene_id[scene_id]
        scene = self.scenes_by_scene_id[scene_id]
        frame_indices = pair.test_indices
        neighbor_indices = pair.neighbor_indices
        num_frames = len(frame_indices)

        item = {}

        # Load encoded prompt
        encoded_prompt, caption = load_encoded_prompt(
            scene.prompt_paths,
            self.generator,
        )
        item["encoded_prompt"] = encoded_prompt
        item["prompt"] = caption

        # Load rendered images
        rgb_rendered = load_indexed_frames(scene.render_dir, frame_indices, filename_format="{:05d}.png")
        item["rgb_rendered"] = resize_to_multiple_of_16(rgb_rendered)
        output_size_source = rgb_rendered

        # Load ground truth images
        if scene.has_gt:
            gt_file_paths = [transforms["frames"][x]["file_path"] for x in frame_indices]
            rgb_gt = load_frames_from_prepared_paths(scene.image_root, gt_file_paths)
            item["rgb_gt"] = resize_to_multiple_of_16(rgb_gt)
            output_size_source = rgb_gt

        item["target_h"] = output_size_source.shape[-2]
        item["target_w"] = output_size_source.shape[-1]

        # Load neighbor frames
        neighbor_file_paths = [transforms["frames"][x]["file_path"] for x in neighbor_indices]
        rgb_neighbors = load_frames_from_prepared_paths(scene.image_root, neighbor_file_paths)
        item["rgb_neighbors"] = resize_to_multiple_of_16(rgb_neighbors)

        # Load opacity
        opacity = load_indexed_frames(scene.opacity_dir, frame_indices, filename_format="{:05d}.png", grayscale=True)
        item["opacity"] = resize_to_multiple_of_16(opacity).squeeze(1)

        # Camera rays condition the target frames, so match the rendered target tensor.
        H, W = item["rgb_rendered"].shape[-2], item["rgb_rendered"].shape[-1]
        camera_items = compute_camera_rays(
            transforms=transforms,
            frame_indices=list(frame_indices),
            neighbor_indices=neighbor_indices,
            scale=scene.camera_scale,
            image_shape=(H, W),
            skip_vae_check=True,  # Eval script handles padding
        )
        item.update(camera_items)

        # Add metadata
        item["frame_indices"] = torch.tensor(frame_indices, dtype=torch.long)
        item["neighbor_indices"] = torch.tensor(neighbor_indices, dtype=torch.long)
        item["scene_id"] = scene_id
        item["chunk_idx"] = pair.chunk_idx
        item["valid_frames_mask"] = torch.ones(num_frames, dtype=torch.bool)

        # is_test_frame: which frames are actual test frames (for metrics)
        if pair.is_test_frame is not None:
            item["is_test_frame"] = torch.tensor(pair.is_test_frame, dtype=torch.bool)
        else:
            item["is_test_frame"] = torch.ones(num_frames, dtype=torch.bool)

        if scene.has_gt:
            gt_index = [int(idx) if bool(keep) else -1 for idx, keep in zip(frame_indices, item["is_test_frame"])]
            item["gt_index"] = torch.tensor(gt_index, dtype=torch.long)
        return item

    def visualize_splits(self, output_path: str | Path | None = None, title: str | None = None) -> None:
        """Visualize train/test splits for all scenes."""

        def get_total_frames(scene_id: str) -> int:
            return len(self.transforms_by_scene_id[scene_id]["frames"])

        visualize_inference_pairs(
            inference_items=self.inference_items,
            get_total_frames=get_total_frames,
            output_path=output_path,
            title=title
            or f"Reconstructed COLMAP - {self.neighbor_selection_mode.value.upper()}: Train (green) / Test (blue)",
            selection_mode=self.neighbor_selection_mode,
        )
