# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Any

import torch

from model_training.data.utils import compute_camera_conditioning


class DatasetBase(torch.utils.data.Dataset, ABC):

    def __init__(
        self,
        dataset_scaling_factor: float,  # make sure magnitude of translations does not get too large
    ):
        self.dataset_scaling_factor = dataset_scaling_factor

    def add_camera_information(
        self,
        item: dict[str, Any],
        intrinsics: dict[str, Any],
        c2ws_world: torch.Tensor,
        neighbor_c2ws_world: torch.Tensor,
        scale: float,
        image_width: int,
        image_height: int,
        skip_vae_check: bool = False,
    ) -> None:
        item.update(
            compute_camera_conditioning(
                intrinsics=intrinsics,
                c2ws_world=c2ws_world,
                neighbor_c2ws_world=neighbor_c2ws_world,
                scale=scale,
                image_height=image_height,
                image_width=image_width,
                skip_vae_check=skip_vae_check,
            )
        )
