# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from model_eval.datasets import reconstructed_colmap_eval
from model_eval.datasets.reconstructed_colmap_eval import ReconstructedColmapEvalDataset
from model_training.data.utils import NeighborSelectionMode


def write_image(path: Path, size: tuple[int, int], mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size).save(path)


class ReconstructedColmapEvalDatasetTests(unittest.TestCase):
    def test_camera_rays_use_render_resolution_when_context_images_are_smaller(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        image_root = root / "image_root"
        render_dir = root / "renders"
        opacity_dir = root / "opacity"

        write_image(image_root / "images/frame_00000.png", (32, 16))
        write_image(image_root / "images/frame_00001.png", (32, 16))
        write_image(render_dir / "00001.png", (64, 32))
        write_image(opacity_dir / "00001.png", (64, 32), mode="L")

        transform_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        (root / "transforms.json").write_text(
            json.dumps(
                {
                    "w": 64,
                    "h": 32,
                    "fl_x": 40.0,
                    "fl_y": 40.0,
                    "cx": 32.0,
                    "cy": 16.0,
                    "frames": [
                        {"file_path": "images/frame_00000.png", "transform_matrix": transform_matrix},
                        {"file_path": "images/frame_00001.png", "transform_matrix": transform_matrix},
                    ],
                }
            )
        )
        (root / "selected_indices.json").write_text("[0]")
        (root / "caption.h5").write_bytes(b"placeholder")
        (root / "split.json").write_text(
            json.dumps(
                {
                    "test": {
                        "scene": {
                            "transforms_path": "transforms.json",
                            "image_root": "image_root",
                            "render_dir": "renders",
                            "opacity_dir": "opacity",
                            "selected_indices_path": "selected_indices.json",
                            "prompt_path": "caption.h5",
                            "camera_scale": 1.0,
                        }
                    }
                }
            )
        )

        def fake_compute_camera_rays(*, image_shape: tuple[int, int], **kwargs):
            h, w = image_shape
            return {"camera_rays": torch.zeros(1, h, w, 6)}

        with (
            patch.object(
                reconstructed_colmap_eval,
                "load_encoded_prompt",
                return_value=(torch.zeros(1, 1, dtype=torch.bfloat16), ""),
            ),
            patch.object(reconstructed_colmap_eval, "compute_camera_rays", side_effect=fake_compute_camera_rays),
        ):
            dataset = ReconstructedColmapEvalDataset(
                split="test",
                split_path=root / "split.json",
                num_views=1,
                neighbor_selection_mode=NeighborSelectionMode.CONSECUTIVE,
                max_test_frames=1,
            )
            item = dataset[0]

        self.assertEqual(tuple(item["rgb_neighbors"].shape[-2:]), (16, 32))
        self.assertEqual(tuple(item["rgb_rendered"].shape[-2:]), (32, 64))
        self.assertEqual(tuple(item["camera_rays"].shape[-3:-1]), tuple(item["rgb_rendered"].shape[-2:]))


if __name__ == "__main__":
    unittest.main()
