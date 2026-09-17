# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DL3DV HDF5 Evaluation Dataset

Dataset for evaluating on DL3DV test scenes using pre-computed HDF5 files.
Train/test indices are read from the HDF5 file's selected_indices field.
Rendered images are also read from the HDF5 file.
"""

import io
from pathlib import Path
from typing import Any, List, Tuple

import av
import h5py
import numpy as np
import torch

from model_training.data.dl3dv_test import DL3DVPairedDatasetTest
from model_training.data.reconstruction_path import parse_reconstruction_hdf5_path
from model_training.data.utils import (
    InferencePair,
    NeighborSelectionMode,
    generate_inference_pairs,
    visualize_inference_pairs,
)


def _count_video_frames(video_bytes: bytes) -> int:
    with av.open(io.BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return stream.frames
        return sum(1 for _ in container.decode(video=0))


class DL3DVHDF5EvalDataset(DL3DVPairedDatasetTest):
    """
    Dataset for evaluating on DL3DV using pre-computed HDF5 files.

    Train IDs come from HDF5's selected_indices field.
    Rendered images are read from the HDF5 file.
    Test frames are the complement of train indices.
    """

    def __init__(
        self,
        split: str,
        split_path: Path,
        dl3dv_dir: Path,
        prompt_dir: Path,
        num_frames: int | None,
        num_views: int,
        dataset_scaling_factor: float,
        frames_per_block: int = 7,
        neighbor_selection_mode: NeighborSelectionMode = NeighborSelectionMode.COVISIBILITY,
        max_test_frames: int | None = None,
        include_all_frames: bool = False,
        filter_scene_id: str | None = None,
        generator: torch.Generator | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            split=split,
            split_path=split_path,
            dl3dv_dir=dl3dv_dir,
            prompt_dir=prompt_dir,
            num_frames=num_frames,
            frames_per_block=frames_per_block,
            num_views=num_views,
            dataset_scaling_factor=dataset_scaling_factor,
            max_neighbors=num_views,
            generator=generator,
            verbose=verbose,
            start_index=0,
            skip_vae_check=include_all_frames or max_test_frames is None,
        )

        self.num_views = num_views
        self.neighbor_selection_mode = neighbor_selection_mode
        self.inference_items: List[Tuple[int, InferencePair]] = []
        filtered_scene_ids = set()

        for data_idx, data_dict in enumerate(self.data):
            if num_views not in data_dict:
                raise ValueError(f"num_views {num_views} not found in data_dict (available: {list(data_dict.keys())})")
            hdf5_path, _ = data_dict[num_views]
            scene_id, half, n_views = parse_reconstruction_hdf5_path(hdf5_path)

            if n_views != num_views:
                continue

            if half != "0":
                continue

            if filter_scene_id is not None and scene_id != filter_scene_id:
                continue

            filtered_scene_ids.add(scene_id)

            with h5py.File(hdf5_path, "r") as f:
                source_train_ids = set(f["selected_indices"][:].tolist())
                render_total_frames = _count_video_frames(f["render_video"][:].tobytes())

            transforms = self.transforms_by_scene_id[scene_id]
            total_frames = min(len(transforms["frames"]), render_total_frames)
            train_ids = {frame_idx for frame_idx in source_train_ids if frame_idx < total_frames}
            if verbose and total_frames != len(transforms["frames"]):
                print(
                    f"Scene {scene_id}: limiting eval frames to {total_frames} HDF5 renders "
                    f"(transforms has {len(transforms['frames'])} frames)"
                )

            extrinsics_c2w = None
            if neighbor_selection_mode == NeighborSelectionMode.COVISIBILITY:
                extrinsics_c2w = np.array([transforms["frames"][i]["transform_matrix"] for i in range(total_frames)])

            pairs = generate_inference_pairs(
                train_ids=train_ids,
                total_frames=total_frames,
                num_train_context=num_views,
                max_test_frames=max_test_frames,
                selection_mode=neighbor_selection_mode,
                extrinsics_c2w=extrinsics_c2w,
                include_all_frames=include_all_frames,
            )

            for pair in pairs:
                pair.scene_id = scene_id
                self.inference_items.append((data_idx, pair))

        if verbose:
            print(
                f"Generated {len(self.inference_items)} inference items "
                f"across {len(filtered_scene_ids)} matched scene(s) "
                f"(base split contains {len(self.data)} scene entries)"
            )

    def __len__(self) -> int:
        return len(self.inference_items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data_idx, pair = self.inference_items[idx]
        item = self.get_item_inner(
            data_idx,
            self.num_views,
            len(pair.neighbor_indices),
            override_frame_indices=pair.test_indices,
            override_neighbor_indices=pair.neighbor_indices,
        )
        item["frame_indices"] = torch.tensor(pair.test_indices, dtype=torch.long)
        item["neighbor_indices"] = torch.tensor(pair.neighbor_indices, dtype=torch.long)
        item["scene_id"] = pair.scene_id
        item["chunk_idx"] = pair.chunk_idx
        # valid_frames_mask: True for actual frames, False for VAE padding (added later by eval script)
        num_total_frames = item["rgb_gt"].shape[0]
        num_inference_frames = len(pair.test_indices)
        valid_mask = torch.zeros(num_total_frames, dtype=torch.bool)
        valid_mask[:num_inference_frames] = True
        item["valid_frames_mask"] = valid_mask

        # Pad is_test_frame to match num_total_frames (base class may pad frames)
        if pair.is_test_frame is not None:
            is_test = torch.tensor(pair.is_test_frame, dtype=torch.bool)
        else:
            is_test = torch.ones(num_inference_frames, dtype=torch.bool)
        # Pad with False for any extra frames added by base class
        if len(is_test) < num_total_frames:
            padding = torch.zeros(num_total_frames - len(is_test), dtype=torch.bool)
            is_test = torch.cat([is_test, padding])
        item["is_test_frame"] = is_test

        gt_index = torch.full((num_total_frames,), -1, dtype=torch.long)
        for frame_i, (frame_idx, keep) in enumerate(zip(pair.test_indices, is_test[:num_inference_frames])):
            if bool(keep):
                gt_index[frame_i] = int(frame_idx)
        item["gt_index"] = gt_index
        return item

    def visualize_splits(self, output_path: str | Path | None = None, title: str | None = None) -> None:
        """Visualize train/test splits for this dataset."""

        def get_total_frames(data_idx: int) -> int:
            hdf5_path, _ = next(iter(self.data[data_idx].values()))
            scene_id, _, _ = parse_reconstruction_hdf5_path(hdf5_path)
            return len(self.transforms_by_scene_id[scene_id]["frames"])

        visualize_inference_pairs(
            inference_items=self.inference_items,
            get_total_frames=get_total_frames,
            output_path=output_path,
            title=title or f"Ours - {self.neighbor_selection_mode.value.upper()}: Train (green) / Test (blue)",
            selection_mode=self.neighbor_selection_mode,
        )
