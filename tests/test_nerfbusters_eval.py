# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from data_processing.nerfbusters import (
    nerfbusters_downsample_factor,
    nerfbusters_gt_relpath,
    nerfbusters_image_folder,
)


@dataclass
class InferencePair:
    neighbor_indices: list[int]
    test_indices: list[int]
    reversed: bool = False
    chunk_idx: int = 0
    scene_id: str | None = None
    is_test_frame: list[bool] | None = None


class NeighborSelectionMode:
    EVENLY_SPACED = "evenly_spaced"
    COVISIBILITY = "covisibility"


@contextmanager
def patched_nerfbusters_eval_dataset():
    utils_module = types.ModuleType("model_training.data.utils")
    utils_module.InferencePair = InferencePair
    utils_module.NeighborSelectionMode = NeighborSelectionMode
    utils_module.compute_camera_rays = lambda *args, **kwargs: {}
    utils_module.generate_inference_pairs = lambda *args, **kwargs: []
    utils_module.load_encoded_prompt = lambda *args, **kwargs: (None, "")
    utils_module.load_indexed_frames = lambda *args, **kwargs: None
    utils_module.resize_to_multiple_of_16 = lambda value: value

    torch_module = types.ModuleType("torch")
    torch_utils_module = types.ModuleType("torch.utils")
    torch_data_module = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    torch_data_module.Dataset = Dataset
    torch_utils_module.data = torch_data_module
    torch_module.utils = torch_utils_module
    torch_module.Generator = object
    torch_module.Tensor = object

    old_utils = sys.modules.get("model_training.data.utils")
    old_torch = sys.modules.get("torch")
    old_torch_utils = sys.modules.get("torch.utils")
    old_torch_data = sys.modules.get("torch.utils.data")
    old_dataset = sys.modules.pop("model_eval.datasets.nerfbusters_eval", None)
    sys.modules["model_training.data.utils"] = utils_module
    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = torch_utils_module
    sys.modules["torch.utils.data"] = torch_data_module
    try:
        module = importlib.import_module("model_eval.datasets.nerfbusters_eval")
        yield module.NerfbustersEvalDataset
    finally:
        sys.modules.pop("model_eval.datasets.nerfbusters_eval", None)
        if old_dataset is not None:
            sys.modules["model_eval.datasets.nerfbusters_eval"] = old_dataset
        if old_utils is None:
            sys.modules.pop("model_training.data.utils", None)
        else:
            sys.modules["model_training.data.utils"] = old_utils
        for module_name, old_module in (
            ("torch", old_torch),
            ("torch.utils", old_torch_utils),
            ("torch.utils.data", old_torch_data),
        ):
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module


class NerfbustersEvalTests(unittest.TestCase):
    def test_shared_resolution_mapping_matches_paper_eval_resolution(self):
        self.assertEqual(nerfbusters_downsample_factor("aloe"), 2)
        self.assertEqual(nerfbusters_image_folder("aloe"), "images_2")
        self.assertEqual(nerfbusters_gt_relpath("aloe", "images/frame_00000.png"), "images_2/frame_00000.png")
        self.assertEqual(nerfbusters_downsample_factor("flowers"), 1)
        self.assertEqual(nerfbusters_image_folder("flowers"), "images")
        self.assertEqual(nerfbusters_gt_relpath("flowers", "images/frame_00000.png"), "images/frame_00000.png")

    def test_dataset_uses_shared_resolution_mapping_by_default(self):
        with patched_nerfbusters_eval_dataset() as NerfbustersEvalDataset:
            dataset = object.__new__(NerfbustersEvalDataset)
            dataset.image_folder = None

            self.assertEqual(dataset._get_image_folder_for_scene("car"), "images_2")
            self.assertEqual(dataset._get_image_folder_for_scene("flowers"), "images")

            dataset.image_folder = "images_4"
            self.assertEqual(dataset._get_image_folder_for_scene("car"), "images_4")

    def test_train_test_split_uses_current_nerfbusters_frame_names(self):
        with patched_nerfbusters_eval_dataset() as NerfbustersEvalDataset:
            dataset = object.__new__(NerfbustersEvalDataset)
            transforms = {
                "frames": [
                    {"file_path": "images/frame_00000.png"},
                    {"file_path": "images/frame_1_00001.png"},
                    {"file_path": "images/frame_00002.png"},
                ]
            }

            train_ids, test_ids = dataset._get_train_test_split(transforms)

        self.assertEqual(train_ids, {0, 2})
        self.assertEqual(test_ids, {1})

    def test_inference_pair_is_split_by_frame_number_gaps(self):
        with patched_nerfbusters_eval_dataset() as NerfbustersEvalDataset:
            dataset = object.__new__(NerfbustersEvalDataset)
        transforms = {
            "frames": [
                {"file_path": "images/frame_1_00010.png"},
                {"file_path": "images/frame_1_00011.png"},
                {"file_path": "images/frame_1_00020.png"},
            ]
        }
        pair = InferencePair(
            neighbor_indices=[0],
            test_indices=[2, 0, 1],
            reversed=False,
            chunk_idx=3,
            scene_id="art",
            is_test_frame=[True, True, True],
        )

        split_pairs = dataset._split_pair_by_frame_number_gaps(transforms, pair)

        self.assertEqual([[0, 1], [2]], [list(p.test_indices) for p in split_pairs])
        self.assertEqual([3000, 3001], [p.chunk_idx for p in split_pairs])

    def test_nerfbusters_output_suffix_separates_3dplus_from_direct_eval(self):
        inference_source = Path("model_eval/run_inference.py").read_text()
        metrics_source = Path("model_eval/compute_metrics_nerfbusters.py").read_text()

        self.assertIn('output_suffix = f"_{args.output_suffix}" if args.output_suffix else ""', inference_source)
        self.assertIn("{trajectory_suffix}{output_suffix}", inference_source)
        self.assertIn('suffix = f"_{output_suffix}" if output_suffix else ""', metrics_source)
        self.assertIn("nerfbusters_{num_views}_evenly_spaced{sink_suffix}{suffix}", metrics_source)


if __name__ == "__main__":
    unittest.main()
