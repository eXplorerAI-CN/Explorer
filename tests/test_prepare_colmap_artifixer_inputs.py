# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from data_processing.prepare_colmap_artifixer_inputs import (
    Camera,
    ColmapScene,
    Image,
    scale_colmap_scene_to_images,
)


def write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", size).save(path)


class ColmapInputValidationTests(unittest.TestCase):
    def test_scales_camera_intrinsics_to_match_cleanly_resized_image_file(self) -> None:
        image_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_image(image_dir / "frame_00001.png", (960, 540))
        scene = ColmapScene(
            cameras=[
                Camera(
                    id=1,
                    model="OPENCV",
                    width=3840,
                    height=2160,
                    params=np.array([1856.0, 1848.0, 1920.0, 1080.0, 0.1, 0.2, 0.3, 0.4]),
                )
            ],
            images=[
                Image(
                    id=1,
                    qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                    tvec=np.zeros(3),
                    camera_id=1,
                    name="frame_00001.png",
                    xys=np.array([[384.0, 216.0], [1920.0, 1080.0]]),
                    point3D_ids=np.array([10, -1]),
                )
            ],
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            scaled = scale_colmap_scene_to_images(image_dir, scene)

        camera = scaled.cameras[0]
        image = scaled.images[0]
        self.assertEqual(camera.width, 960)
        self.assertEqual(camera.height, 540)
        np.testing.assert_allclose(camera.params, [464.0, 462.0, 480.0, 270.0, 0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(image.xys, [[96.0, 54.0], [480.0, 270.0]])
        np.testing.assert_array_equal(image.point3D_ids, [10, -1])
        self.assertIn("Scaling COLMAP camera 1 from 3840x2160 to 960x540", stdout.getvalue())

    def test_rejects_camera_resolution_mismatch_that_changes_aspect_ratio(self) -> None:
        image_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_image(image_dir / "frame_00001.png", (32, 20))
        scene = ColmapScene(
            cameras=[
                Camera(
                    id=1,
                    model="OPENCV",
                    width=64,
                    height=32,
                    params=np.array([40.0, 40.0, 32.0, 16.0, 0.0, 0.0, 0.0, 0.0]),
                )
            ],
            images=[
                Image(
                    id=1,
                    qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                    tvec=np.zeros(3),
                    camera_id=1,
                    name="frame_00001.png",
                    xys=np.empty((0, 2)),
                    point3D_ids=np.empty((0,)),
                )
            ],
        )

        with self.assertRaisesRegex(AssertionError, "camera/image size mismatch"):
            scale_colmap_scene_to_images(image_dir, scene)

    def test_mismatch_error_reports_the_image_for_that_camera(self) -> None:
        image_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_image(image_dir / "camera_2.png", (32, 20))
        write_image(image_dir / "camera_1.png", (32, 16))
        scene = ColmapScene(
            cameras=[
                Camera(
                    id=1,
                    model="OPENCV",
                    width=64,
                    height=32,
                    params=np.array([40.0, 40.0, 32.0, 16.0, 0.0, 0.0, 0.0, 0.0]),
                ),
                Camera(
                    id=2,
                    model="OPENCV",
                    width=64,
                    height=32,
                    params=np.array([40.0, 40.0, 32.0, 16.0, 0.0, 0.0, 0.0, 0.0]),
                ),
            ],
            images=[
                Image(
                    id=2,
                    qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                    tvec=np.zeros(3),
                    camera_id=2,
                    name="camera_2.png",
                    xys=np.empty((0, 2)),
                    point3D_ids=np.empty((0,)),
                ),
                Image(
                    id=1,
                    qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                    tvec=np.zeros(3),
                    camera_id=1,
                    name="camera_1.png",
                    xys=np.empty((0, 2)),
                    point3D_ids=np.empty((0,)),
                ),
            ],
        )

        with self.assertRaisesRegex(AssertionError, "camera_2.png"):
            scale_colmap_scene_to_images(image_dir, scene)


if __name__ == "__main__":
    unittest.main()
