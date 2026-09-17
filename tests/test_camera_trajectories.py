# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_processing.camera_trajectories import (
    CAMERA_CONVENTION_FLIP,
    applied_transform_matrix,
    assert_target_only_trajectory,
    opencv_w2c_to_opengl_c2w,
    opengl_c2w_to_opencv_w2c,
    read_camera_trajectory,
    transforms_json,
)


def rigid_pose(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return pose


class CameraTrajectoryConventionTests(unittest.TestCase):
    def test_colmap_pose_round_trips_through_transforms_convention(self) -> None:
        world_to_camera = rigid_pose(
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
            (1.0, 2.0, 3.0),
        )

        transform_matrix = opencv_w2c_to_opengl_c2w(world_to_camera)
        roundtrip_world_to_camera = opengl_c2w_to_opencv_w2c(transform_matrix)

        np.testing.assert_allclose(roundtrip_world_to_camera, world_to_camera)

    def test_materialized_colmap_pose_matches_3dgrut_transforms_renderer(self) -> None:
        transform_matrix = rigid_pose(
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
            (4.0, 5.0, 6.0),
        )
        applied_transform = rigid_pose(
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64),
            (0.5, 1.5, 2.5),
        )

        materialized_world_to_camera = opengl_c2w_to_opencv_w2c(transform_matrix, applied_transform)
        expected_renderer_c2w = np.linalg.inv(
            CAMERA_CONVENTION_FLIP @ np.linalg.inv(transform_matrix) @ applied_transform
        )

        np.testing.assert_allclose(np.linalg.inv(materialized_world_to_camera), expected_renderer_c2w)

    def test_read_camera_trajectory_removes_applied_transform_without_changing_render_pose(self) -> None:
        transform_matrix = rigid_pose(
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
            (4.0, 5.0, 6.0),
        )
        applied_transform = rigid_pose(
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64),
            (0.5, 1.5, 2.5),
        )
        trajectory = {
            "camera_model": "OPENCV",
            "w": 640,
            "h": 480,
            "fl_x": 500.0,
            "fl_y": 510.0,
            "cx": 320.0,
            "cy": 240.0,
            "applied_transform": applied_transform[:3].tolist(),
            "frames": [
                {
                    "transform_matrix": transform_matrix.tolist(),
                    "w": 800,
                    "h": 600,
                    "fl_x": 700.0,
                    "fl_y": 710.0,
                    "cx": 400.0,
                    "cy": 300.0,
                }
            ],
        }

        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "trajectory.json"
        path.write_text(json.dumps(trajectory))

        normalized = read_camera_trajectory(path)

        self.assertNotIn("applied_transform", normalized)
        self.assertEqual(normalized["frames"][0]["w"], 800)
        self.assertEqual(normalized["frames"][0]["h"], 600)
        self.assertEqual(normalized["frames"][0]["fl_x"], 700.0)
        self.assertEqual(normalized["frames"][0]["fl_y"], 710.0)
        original_world_to_camera = opengl_c2w_to_opencv_w2c(transform_matrix, applied_transform)
        normalized_world_to_camera = opengl_c2w_to_opencv_w2c(normalized["frames"][0]["transform_matrix"])
        np.testing.assert_allclose(normalized_world_to_camera, original_world_to_camera)

    def test_applied_transform_accepts_nerfstudio_3x4_matrix(self) -> None:
        transforms = {"applied_transform": [[1, 0, 0, 3], [0, 1, 0, 4], [0, 0, 1, 5]]}

        np.testing.assert_allclose(applied_transform_matrix(transforms)[3], [0, 0, 0, 1])

    def test_transforms_json_keeps_intrinsics_without_extra_metadata(self) -> None:
        transforms = transforms_json(
            {
                "camera_model": "OPENCV",
                "w": 640,
                "h": 480,
                "fl_x": 500.0,
                "fl_y": 510.0,
                "cx": 320.0,
                "cy": 240.0,
            },
            [{"transform_matrix": np.eye(4).tolist()}],
        )

        self.assertEqual(transforms["camera_model"], "OPENCV")
        self.assertNotIn("transforms_pose_convention", transforms)

    def test_target_only_trajectory_rejects_source_image_frames(self) -> None:
        trajectory = {
            "frames": [
                {"transform_matrix": np.eye(4).tolist()},
                {"file_path": "images/0102.jpg", "transform_matrix": np.eye(4).tolist()},
            ]
        }

        with self.assertRaisesRegex(AssertionError, "target cameras only"):
            assert_target_only_trajectory(trajectory, "demo trajectory")


if __name__ == "__main__":
    unittest.main()
