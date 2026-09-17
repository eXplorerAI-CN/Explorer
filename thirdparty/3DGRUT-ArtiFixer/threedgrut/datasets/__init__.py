# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .dataset_colmap import ColmapDataset
from .dataset_nerf import NeRFDataset
from .dataset_scannetpp import ScannetppDataset


def make(name: str, config, ray_jitter):
    match name:
        case "nerf":
            train_dataset = NeRFDataset(
                config.path,
                split="train",
                bg_color=config.model.background.color,
                ray_jitter=ray_jitter,
            )
            val_dataset = NeRFDataset(
                config.path,
                split="val",
                bg_color=config.model.background.color,
            )
        case "colmap":
            train_dataset = ColmapDataset(
                config.path,
                split="train",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                ray_jitter=ray_jitter,
                selected_indices_file=config.selected_indices_file, # A json file with the first (or second) half ordered camera poses
                num_selected_indices=config.num_selected_indices, # Number of selected camera indices for sparse recon
                train_test_split_file=config.train_test_split_file, # For mipnerf360 ReconFusion, a json file with train-test camera poses
                image_path_override=config.image_path_override, # Override the image path (to distill renderings in training)
            )
            val_dataset = ColmapDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                selected_indices_file=config.selected_indices_file, # A json file with the first (or second) half ordered camera poses
                num_selected_indices=config.num_selected_indices, # Number of selected camera indices for sparse recon
                train_test_split_file=config.train_test_split_file, # For mipnerf360 ReconFusion, a json file with train-test camera poses
                image_path_override=config.image_path_override, # Override the image path (to distill renderings in training)
            )
        case "scannetpp":
            train_dataset = ScannetppDataset(
                config.path,
                split="train",
                ray_jitter=ray_jitter,
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
            val_dataset = ScannetppDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
        case _:
            raise ValueError(
                f'Unsupported dataset type: {config.dataset.type}. Choose between: ["colmap", "nerf", "scannetpp"].'
            )

    return train_dataset, val_dataset


def make_test(name: str, config):
    match name:
        case "nerf":
            dataset = NeRFDataset(
                config.path,
                split="test",
                bg_color=config.model.background.color,
            )
        case "colmap":
            dataset = ColmapDataset(
                config.path,
                split="test", # test mode to ensure we render all the images regardless of the selected indices
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                selected_indices_file=config.selected_indices_file, # A json file with the first (or second) half ordered camera poses
                num_selected_indices=config.num_selected_indices, # Number of selected camera indices for sparse recon
                train_test_split_file=config.train_test_split_file, # For mipnerf360 ReconFusion, a json file with train-test camera poses
            )
        case "scannetpp":
            dataset = ScannetppDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
        case _:
            raise ValueError(
                f'Unsupported dataset type: {config.dataset.type}. Choose between: ["colmap", "nerf", "scannetpp"].'
            )
    return dataset
