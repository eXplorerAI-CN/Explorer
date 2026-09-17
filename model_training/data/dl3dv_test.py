# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import torch

from model_training.data.dl3dv_base import DL3DVPairedDatasetBase


class DL3DVPairedDatasetTest(DL3DVPairedDatasetBase):

    def __init__(
        self,
        split: str,
        split_path: Path,
        dl3dv_dir: Path,
        prompt_dir: Path,
        num_frames: int,
        frames_per_block: int | None,
        num_views: int,
        start_index: int,
        dataset_scaling_factor: float,
        validation_seed: int = 42,
        max_neighbors: int | None = None,
        generator: torch.Generator | None = None,
        verbose: bool = False,
        skip_vae_check: bool = False,
    ):
        super().__init__(
            split=split,
            split_path=split_path,
            dl3dv_dir=dl3dv_dir,
            prompt_dir=prompt_dir,
            num_frames=num_frames,
            frames_per_block=frames_per_block,
            return_unencoded_prompt=True,
            dataset_scaling_factor=dataset_scaling_factor,
            max_neighbors=max_neighbors,
            generator=generator,
            verbose=verbose,
            skip_vae_check=skip_vae_check,
        )

        self.num_views = num_views
        self.start_index = start_index
        self.validation_seed = validation_seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        logical_index = idx + self.start_index
        data_idx = logical_index % len(self.data)
        generator = torch.Generator().manual_seed(self.validation_seed + logical_index)
        return self.get_item_inner(data_idx, self.num_views, self.num_views, generator=generator)
