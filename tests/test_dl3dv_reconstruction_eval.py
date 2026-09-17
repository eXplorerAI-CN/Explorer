# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from model_eval.datasets.dl3dv_reconstruction_eval import DL3DVReconstructionEvalDataset
from model_training.data.utils import NeighborSelectionMode


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


class DL3DVReconstructionEvalTests(unittest.TestCase):
    def test_loads_extracted_benchmark_scene_when_zip_subdir_is_missing(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_id = "scene_a"
        dl3dv_dir = tmp_dir / "DL3DV-ALL-960P"
        transforms_root = dl3dv_dir / scene_id / "nerfstudio"
        split_path = tmp_dir / "split.json"
        recon_root = tmp_dir / "recon"
        prompt_root = tmp_dir / "prompts"
        checkpoint_dir = recon_root / scene_id / "benchmark_distill" / scene_id / "ours_30000"

        _write_json(
            transforms_root / "transforms.json",
            {
                "frames": [
                    {"file_path": "images/frame_00000.png", "transform_matrix": [[1, 0, 0, 0]]},
                    {"file_path": "images/frame_00001.png", "transform_matrix": [[1, 0, 0, 1]]},
                    {"file_path": "images/frame_00002.png", "transform_matrix": [[1, 0, 0, 2]]},
                ]
            },
        )
        (transforms_root / "images_4").mkdir(parents=True)
        _write_json(split_path, {"test": {scene_id: [{"subdir": "2K"}]}})
        _write_json(checkpoint_dir / "selected_indices.json", [0])
        (checkpoint_dir / "renders").mkdir(parents=True)
        (checkpoint_dir / "opacity").mkdir(parents=True)
        for index in range(3):
            (checkpoint_dir / "renders" / f"{index}.png").write_text("render")
            (checkpoint_dir / "opacity" / f"{index}.png").write_text("opacity")
        prompt_dir = prompt_root / "2K" / scene_id
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "prompt.h5").write_text("prompt")

        dataset = DL3DVReconstructionEvalDataset(
            split="test",
            split_path=split_path,
            dl3dv_dir=dl3dv_dir,
            recon_results_dir=recon_root,
            prompt_dir=prompt_root,
            num_views=1,
            dataset_scaling_factor=1.0,
            checkpoint="30000",
            neighbor_selection_mode=NeighborSelectionMode.COVISIBILITY,
            include_all_frames=True,
            recon_subdir="benchmark_distill",
        )

        scene = dataset.scenes_by_scene_id[scene_id]
        self.assertEqual(scene.transforms_path, transforms_root / "transforms.json")
        self.assertEqual(scene.image_root, transforms_root)
        self.assertEqual(dataset.transforms_by_scene_id[scene_id]["frames"][0]["file_path"], "images_4/frame_00000.png")


if __name__ == "__main__":
    unittest.main()
