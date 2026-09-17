# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from model_eval.dl3dv_distill_validation import main, validate_override_frame_indices


class DL3DVDistillValidationTests(unittest.TestCase):
    def test_accepts_override_indices_inside_scene_frame_table(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "scene" / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene" / "frames" / "batch_0000" / "pred"
        scene_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(
            json.dumps({"frames": [{"file_path": "0.png"}, {"file_path": "1.png"}, {"file_path": "2.png"}]})
        )
        (pred_dir / "00000.png").write_text("pred")
        (pred_dir / "00002.png").write_text("pred")

        validate_override_frame_indices(scene_dir, pred_dir)

    def test_rejects_override_indices_outside_scene_frame_table(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "scene" / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene" / "frames" / "batch_0000" / "pred"
        scene_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(
            json.dumps({"frames": [{"file_path": "0.png"}, {"file_path": "1.png"}, {"file_path": "2.png"}]})
        )
        (pred_dir / "00003.png").write_text("pred")

        with self.assertRaisesRegex(ValueError, "outside the scene frame table"):
            validate_override_frame_indices(scene_dir, pred_dir)

    def test_rejects_override_indices_with_shifted_reference_frame_table(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "scene" / "nerfstudio"
        reference_scene_dir = tmp_dir / "reference" / "scene" / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene" / "frames" / "batch_0000" / "pred"
        scene_dir.mkdir(parents=True)
        reference_scene_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(
            json.dumps(
                {"frames": [{"file_path": "images/frame_00001.png"}, {"file_path": "images/frame_00003.png"}]}
            )
        )
        (reference_scene_dir / "transforms.json").write_text(
            json.dumps(
                {"frames": [{"file_path": "images/frame_00001.png"}, {"file_path": "images/frame_00002.png"}]}
            )
        )
        (pred_dir / "00001.png").write_text("pred")

        with self.assertRaisesRegex(ValueError, "different frame at override index"):
            validate_override_frame_indices(
                scene_dir,
                pred_dir,
                reference_scene_dataset_path=reference_scene_dir,
            )

    def test_rejects_override_indices_with_matching_paths_but_different_pose(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "scene" / "nerfstudio"
        reference_scene_dir = tmp_dir / "reference" / "scene" / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene" / "frames" / "batch_0000" / "pred"
        scene_dir.mkdir(parents=True)
        reference_scene_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(
            json.dumps(
                {
                    "frames": [
                        {"file_path": "images/frame_00001.png", "transform_matrix": [[1, 0, 0, 0]]},
                    ]
                }
            )
        )
        (reference_scene_dir / "transforms.json").write_text(
            json.dumps(
                {
                    "frames": [
                        {"file_path": "images/frame_00001.png", "transform_matrix": [[1, 0, 0, 1]]},
                    ]
                }
            )
        )
        (pred_dir / "00000.png").write_text("pred")

        with self.assertRaisesRegex(ValueError, "camera metadata differs"):
            validate_override_frame_indices(scene_dir, pred_dir, reference_scene_dataset_path=reference_scene_dir)

    def test_cli_reports_validation_errors_without_traceback(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "scene" / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene" / "frames" / "batch_0000" / "pred"
        scene_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(
            json.dumps({"frames": [{"file_path": "0.png"}, {"file_path": "1.png"}, {"file_path": "2.png"}]})
        )
        (pred_dir / "00003.png").write_text("pred")
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            status = main([str(scene_dir), str(pred_dir)])

        self.assertEqual(status, 2)
        self.assertIn("outside the scene frame table", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
