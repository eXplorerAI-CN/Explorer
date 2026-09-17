# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import h5py
import numpy as np
import torch
from tqdm import tqdm

from model_training.constants import MAX_SEQUENCE_LENGTH
from model_training.data.dataset_base import DatasetBase
from model_training.data.reconstruction_path import parse_reconstruction_hdf5_path
from model_training.data.scene_utils import load_scene_transforms_from_zip
from model_training.data.utils import load_frames_from_zip, resize_to_multiple_of_16
from model_training.utils.video_io import decode_video_to_frames

# Permanent data quarantine shared with split generation.
BAD_SCENE_IDS: frozenset[str] = frozenset(
    (Path(__file__).resolve().parents[2] / "data_processing" / "bad_scene_ids.txt").read_text().split()
)


def decode_depth_frames(min_val: float, max_val: float, quantized: np.ndarray) -> np.ndarray:
    """Decode quantized depth back to float32."""
    normalized = quantized.astype(np.float32) / 65535.0
    return normalized * (max_val - min_val) + min_val


class DL3DVPairedDatasetBase(DatasetBase):

    def __init__(
        self,
        split: str,
        split_path: Path,
        dl3dv_dir: Path,
        prompt_dir: Path,
        num_frames: int | None,
        frames_per_block: int | None,
        return_unencoded_prompt: bool,
        dataset_scaling_factor: float,  # make sure magnitude of translations does not get too large
        max_neighbors: int | None = None,
        generator: torch.Generator | None = None,
        verbose: bool = False,
        skip_vae_check: bool = False,
    ):
        super().__init__(
            dataset_scaling_factor=dataset_scaling_factor,
        )
        self.dl3dv_dir = dl3dv_dir
        self.skip_vae_check = skip_vae_check
        self.return_unencoded_prompt = return_unencoded_prompt
        self.generator = generator
        self.dl3dv_downsample_factor = 4
        self.frames_per_block = frames_per_block
        # go through all available hdf5 files (one per reconstruction)
        reconstructions_by_scene_id_and_half = defaultdict(dict)
        self.transforms_by_scene_id = {}
        self.prompts_by_scene_id = {}
        self.scene_id_to_subdir = {}
        self.scene_id_zip_has_nested_prefix = {}
        if num_frames is not None and num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        self.num_frames = num_frames  # None during auto-discovery, resolved after scene loop

        with split_path.open() as f:
            reconstructions_by_scene_id = json.load(f)[split]

        for scene_id, reconstructions in tqdm(reconstructions_by_scene_id.items(), disable=not verbose):
            if scene_id in BAD_SCENE_IDS:
                continue
            subdir = reconstructions[0]["subdir"]
            scene_prompt_dir = prompt_dir / subdir / scene_id

            with ZipFile(self.dl3dv_dir / subdir / f"{scene_id}.zip", "r") as zf:
                transforms, scene_root = load_scene_transforms_from_zip(zf, scene_id)
                has_nested_prefix = bool(scene_root)

            scene_num_frames = len(transforms["frames"])
            if num_frames is None:
                self.num_frames = max(self.num_frames or 0, scene_num_frames)

            if num_frames is not None and scene_num_frames < num_frames:
                if verbose:
                    print(
                        f"Skipping scene {scene_id} because it has less than {num_frames} frames ({scene_num_frames})"
                    )
                continue

            # Start with frame stride one since others might be too choppy
            prompt_paths = sorted(
                scene_prompt_dir.glob(f"frames_{num_frames if num_frames is not None else 'all'}_stride_1*.h5")
            )
            if len(prompt_paths) == 0:
                if verbose:
                    print(f"WARNING: No prompt files found for scene {scene_id} in {scene_prompt_dir}")
                continue

            self.scene_id_to_subdir[scene_id] = subdir
            self.transforms_by_scene_id[scene_id] = transforms
            self.prompts_by_scene_id[scene_id] = prompt_paths
            self.scene_id_zip_has_nested_prefix[scene_id] = has_nested_prefix

            for reconstruction in reconstructions:
                num_selected_indices = int(reconstruction["num_selected_indices"])
                if max_neighbors is not None and num_selected_indices > max_neighbors:
                    continue
                hdf5_path = Path(reconstruction["hdf5_path"])
                # Explicit raise (not assert) so validation survives `python -O`.
                if not hdf5_path.exists():
                    raise FileNotFoundError(f"HDF5 file {hdf5_path} does not exist")
                scale = float(reconstruction["scale"])
                if not (math.isfinite(scale) and scale > 0):
                    raise ValueError(f"HDF5 file {hdf5_path} has scale {scale} in split JSON; expected finite > 0")
                recon_key = (scene_id, reconstruction["scene_half"])
                reconstructions_by_scene_id_and_half[recon_key][num_selected_indices] = (hdf5_path, scale)

        self.available_splits: tuple[int, ...] = ()
        canonical_key: tuple[str, str] | None = None
        self.data = []
        for key, recons in sorted(reconstructions_by_scene_id_and_half.items()):
            splits = tuple(sorted(recons.keys()))
            if canonical_key is None:
                self.available_splits = splits
                canonical_key = key
            elif splits != self.available_splits:
                raise ValueError(
                    f"Heterogeneous reconstruction splits: scene-half {key} has {splits}, "
                    f"but {canonical_key} established {self.available_splits}"
                )
            self.data.append(recons)

        if verbose:
            print(f"Found {len(self.data)} data items with splits {self.available_splits}")

        if self.num_frames is None or self.num_frames <= 0:
            raise ValueError("No scenes found with frames - cannot determine num_frames")
        if num_frames is None:
            # Auto-discovered max frame count: round up to satisfy VAE and AR constraints.
            vae_acceptable = math.ceil((self.num_frames - 1) / 4) * 4 + 1
            if self.frames_per_block is not None:
                num_blocks = (vae_acceptable - 1) // 4 + 1
                ar_blocks = math.ceil(num_blocks / self.frames_per_block) * self.frames_per_block
                self.num_frames = (ar_blocks - 1) * 4 + 1
            else:
                self.num_frames = vae_acceptable
        else:
            # Explicitly provided: strict validation; the user should pass a valid value.
            if (self.num_frames - 1) % 4 != 0:
                raise ValueError(
                    f"(num_frames - 1) must be divisible by 4 (VAE temporal scale factor), "
                    f"but got num_frames={self.num_frames}. Try {(self.num_frames - 1) // 4 * 4 + 1} or "
                    f"{((self.num_frames - 1) // 4 + 1) * 4 + 1}."
                )
            if self.frames_per_block is not None:
                latent_frames = (self.num_frames - 1) // 4 + 1
                if latent_frames % self.frames_per_block != 0:
                    raise ValueError(
                        f"Latent frame count must be divisible by frames_per_block={self.frames_per_block}, "
                        f"but got {latent_frames} latent frames (from num_frames={self.num_frames})."
                    )
        latent_frames = (self.num_frames - 1) // 4 + 1
        if verbose:
            print(f"Using {self.num_frames} frames ({latent_frames} latent frames)")

    def get_item_inner(
        self,
        data_idx: int,
        num_train_views: int,
        num_neighbors: int,
        override_frame_indices: list[int] | None = None,
        override_neighbor_indices: list[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        generator = self.generator if generator is None else generator
        hdf5_path, scale = self.data[data_idx][num_train_views]

        scene_id, _, _ = parse_reconstruction_hdf5_path(hdf5_path)
        transforms = self.transforms_by_scene_id[scene_id]

        item = dict()

        prompt_path = self.prompts_by_scene_id[scene_id][
            torch.randint(0, len(self.prompts_by_scene_id[scene_id]), size=(1,), generator=generator)
        ]
        with h5py.File(prompt_path, "r") as f:
            start_frame_candidates = list(f.keys())
            prompt_dataset = f[
                start_frame_candidates[torch.randint(0, len(start_frame_candidates), size=(1,), generator=generator)]
            ]
            frame_indices = (
                override_frame_indices if override_frame_indices is not None else prompt_dataset.attrs["image_indices"]
            )
            if override_frame_indices is None:
                original_frame_indices = frame_indices
                index = torch.randint(0, len(original_frame_indices), size=(1,), generator=generator)
                direction = torch.randint(0, 2, size=(1,), generator=generator) * 2 - 1
                frame_indices = []
                while len(frame_indices) < self.num_frames:
                    frame_indices.append(original_frame_indices[index])
                    if index == 0:
                        direction = 1
                    elif index == len(original_frame_indices) - 1:
                        direction = -1
                    index += direction

            # Numpy doesn't have bfloat16 data type, so convert from uint16
            encoded_prompt = torch.tensor(prompt_dataset[:], dtype=torch.uint16).view(torch.bfloat16)
            encoded_prompt = torch.cat(
                [
                    encoded_prompt,
                    torch.zeros(
                        MAX_SEQUENCE_LENGTH - encoded_prompt.shape[0],
                        encoded_prompt.shape[1],
                        dtype=torch.bfloat16,
                    ),
                ],
                dim=0,
            )
            item["encoded_prompt"] = encoded_prompt

            if self.return_unencoded_prompt:
                item["prompt"] = prompt_dataset.attrs["caption"]
                item["prompt_file"] = f"{prompt_path.parent.name}/{prompt_path.name}"

        scale = scale * self.dataset_scaling_factor
        try:
            with h5py.File(hdf5_path, "r") as f:
                training_indices = f["selected_indices"][:]
                rendered_frames = decode_video_to_frames(f["render_video"][:].tobytes())[frame_indices]
                opacity_frames = decode_video_to_frames(f["opacity_video"][:].tobytes())[frame_indices][..., :1]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load reconstruction data from {hdf5_path} "
                f"(scene_id={scene_id}, num_train_views={num_train_views})"
            ) from e

        rgb_rendered = torch.FloatTensor(rendered_frames).permute(0, 3, 1, 2) / 255.0
        item["rgb_rendered"] = resize_to_multiple_of_16(rgb_rendered)

        rgb_gt_file_paths = [transforms["frames"][x]["file_path"] for x in frame_indices]
        if override_neighbor_indices is not None:
            neighbor_indices = np.array(override_neighbor_indices)
        else:
            neighbor_indices = training_indices[
                torch.multinomial(
                    torch.ones(len(training_indices)), num_neighbors, replacement=False, generator=generator
                ).numpy()
            ]
        neighbor_file_paths = [transforms["frames"][x]["file_path"] for x in neighbor_indices]

        has_nested_prefix = self.scene_id_zip_has_nested_prefix[scene_id]
        with ZipFile(self.dl3dv_dir / self.scene_id_to_subdir[scene_id] / f"{scene_id}.zip", "r") as zf:
            rgb_gt = load_frames_from_zip(
                zf, scene_id, rgb_gt_file_paths, has_nested_prefix, self.dl3dv_downsample_factor
            )
            rgb_neighbors = load_frames_from_zip(
                zf, scene_id, neighbor_file_paths, has_nested_prefix, self.dl3dv_downsample_factor
            )

        item["target_h"] = rgb_gt.shape[-2]
        item["target_w"] = rgb_gt.shape[-1]
        item["rgb_gt"] = resize_to_multiple_of_16(rgb_gt)
        item["rgb_neighbors"] = resize_to_multiple_of_16(rgb_neighbors)
        opacity = torch.FloatTensor(opacity_frames).permute(0, 3, 1, 2) / 255.0
        item["opacity"] = resize_to_multiple_of_16(opacity).squeeze(1)

        self.add_camera_information(
            item,
            transforms,
            torch.stack([torch.FloatTensor(transforms["frames"][idx]["transform_matrix"]) for idx in frame_indices]),
            torch.stack([torch.FloatTensor(transforms["frames"][idx]["transform_matrix"]) for idx in neighbor_indices]),
            scale,
            item["rgb_gt"].shape[-1],  # Use resized dimensions
            item["rgb_gt"].shape[-2],
            skip_vae_check=self.skip_vae_check,
        )

        return item
