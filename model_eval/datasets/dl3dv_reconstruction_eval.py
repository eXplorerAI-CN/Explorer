# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DL3DV reconstruction-render eval paths."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import torch

from model_eval.datasets.reconstructed_colmap_eval import ReconstructedColmapEvalDataset, ReconstructedColmapScene
from model_eval.dl3dv_difix_frame_remap import apply_difix_frame_remap
from model_training.data.scene_utils import downsampled_image_path, load_scene_transforms_from_zip
from model_training.data.utils import NeighborSelectionMode


class DL3DVReconstructionEvalDataset(ReconstructedColmapEvalDataset):
    """Adapt existing DL3DV reconstruction-render layouts to the reconstructed-COLMAP eval loader."""

    def __init__(
        self,
        split: str,
        split_path: Path,
        dl3dv_dir: Path,
        recon_results_dir: Path,
        prompt_dir: Path,
        num_views: int | None,
        dataset_scaling_factor: float,
        image_downsample_factor: int = 4,
        checkpoint: str = "30000",
        neighbor_selection_mode: NeighborSelectionMode = NeighborSelectionMode.COVISIBILITY,
        max_test_frames: int | None = None,
        include_all_frames: bool = False,
        filter_scene_id: str | None = None,
        generator: torch.Generator | None = None,
        verbose: bool = False,
        recon_subdir: str | None = None,
        use_difix_frame_remap: bool = False,
    ):
        self.dl3dv_dir = Path(dl3dv_dir)
        self.recon_results_dir = Path(recon_results_dir)
        self.prompt_dir = Path(prompt_dir)
        self.dataset_scaling_factor = dataset_scaling_factor
        self.image_downsample_factor = image_downsample_factor
        self.checkpoint = checkpoint
        self.recon_subdir = recon_subdir
        self.use_difix_frame_remap = use_difix_frame_remap
        super().__init__(
            split=split,
            split_path=split_path,
            num_views=num_views,
            neighbor_selection_mode=neighbor_selection_mode,
            max_test_frames=max_test_frames,
            include_all_frames=include_all_frames,
            filter_scene_id=filter_scene_id,
            generator=generator,
            verbose=verbose,
        )

    def _get_scene_dir(self, scene_id: str) -> Path:
        if self.recon_subdir is not None:
            return self.recon_results_dir / scene_id / self.recon_subdir / scene_id
        return (
            self.recon_results_dir
            / "difix_split_reconstruction_eval"
            / scene_id
            / "difix_split_reconstruction"
            / scene_id
        )

    def _get_checkpoint_dir(self, scene_id: str) -> Path:
        return self._get_scene_dir(scene_id) / f"ours_{self.checkpoint}"

    @staticmethod
    def _rendered_frame_count(render_dir: Path, opacity_dir: Path) -> int:
        render_indices = sorted(int(path.stem) for path in render_dir.glob("*.png"))
        opacity_indices = sorted(int(path.stem) for path in opacity_dir.glob("*.png"))
        assert render_indices, f"{render_dir} has no rendered frames"
        assert render_indices == opacity_indices, f"{render_dir} and {opacity_dir} frame sets differ"
        assert render_indices == list(range(len(render_indices))), f"{render_dir} frame indices must be contiguous"
        return len(render_indices)

    @staticmethod
    def _scene_items_from_split(split_data: object) -> list[tuple[str, str]]:
        if isinstance(split_data, dict):
            return [(scene_id, recons[0]["subdir"]) for scene_id, recons in split_data.items() if recons]

        scene_items = []
        seen = set()
        assert isinstance(split_data, list), "DL3DV split must be a mapping or list of scene metadata"
        for item in split_data:
            assert isinstance(item, dict), f"DL3DV split item must be a mapping, got {item!r}"
            for metadata in item.values():
                scene_id = metadata["scene_id"]
                if scene_id not in seen:
                    scene_items.append((scene_id, metadata["subdir"]))
                    seen.add(scene_id)
        return scene_items

    def _load_scene_data(
        self,
        split_path: Path,
        split: str,
        filter_scene_id: str | None,
        verbose: bool,
    ) -> None:
        with open(split_path) as f:
            split_data = json.load(f)[split]

        self._reset_scene_data()

        for scene_id, source_subdir in self._scene_items_from_split(split_data):
            if filter_scene_id is not None and scene_id != filter_scene_id:
                continue

            checkpoint_dir = self._get_checkpoint_dir(scene_id)
            render_dir = checkpoint_dir / "renders"
            opacity_dir = checkpoint_dir / "opacity"
            selected_indices_path = checkpoint_dir / "selected_indices.json"
            transforms_path = self.dl3dv_dir / source_subdir / f"{scene_id}.zip"
            prompt_paths = sorted((self.prompt_dir / source_subdir / scene_id).glob("*.h5"))

            required_paths = (render_dir, opacity_dir, selected_indices_path, transforms_path)
            missing = [path for path in required_paths if not path.exists()]
            if not prompt_paths:
                missing.append(self.prompt_dir / source_subdir / scene_id)
            if missing:
                if verbose:
                    print(f"Skipping {scene_id}: missing {missing[0]}")
                continue

            with ZipFile(transforms_path, "r") as zip_file:
                transforms, _ = load_scene_transforms_from_zip(zip_file, scene_id)
            assert isinstance(transforms["frames"], list), f"{transforms_path} must contain a frames list"
            if self.use_difix_frame_remap:
                transforms = apply_difix_frame_remap(transforms, scene_id)
            rendered_frame_count = self._rendered_frame_count(render_dir, opacity_dir)
            if self.use_difix_frame_remap and rendered_frame_count != len(transforms["frames"]):
                raise ValueError(
                    f"Scene {scene_id} has {rendered_frame_count} renders but {len(transforms['frames'])} remapped frames"
                )
            transforms["frames"] = [
                {**frame, "file_path": downsampled_image_path(frame["file_path"], self.image_downsample_factor)}
                for frame in transforms["frames"][:rendered_frame_count]
            ]
            scene = ReconstructedColmapScene(
                scene_id=scene_id,
                transforms_path=transforms_path,
                image_root=transforms_path,
                render_dir=render_dir,
                opacity_dir=opacity_dir,
                selected_indices_path=selected_indices_path,
                target_indices_path=None,
                prompt_paths=prompt_paths,
                camera_scale=self.dataset_scaling_factor,
                has_gt=True,
            )
            self._register_scene(scene, transforms, verbose)

        if verbose:
            print(f"Found {len(self.scene_ids)} DL3DV scenes with reconstruction renders")
