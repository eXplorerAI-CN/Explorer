# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import unittest
from unittest.mock import patch

from scripts.download_dl3dv_scene import (
    DL3DV_DATASET,
    main,
)

SCENE_ID = "15ff83e2531668d27c92091c97d31401ce323e24ee7c844cb32d5109ab9335f7"
SUBDIR = "8K"


class DownloadDl3dvSceneTests(unittest.TestCase):
    def test_downloads_requested_scene_with_huggingface_api(self) -> None:
        with (
            patch(
                "scripts.download_dl3dv_scene.hf_hub_download",
                return_value="/data/DL3DV-ALL-960P/8K/scene_a.zip",
            ) as download,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = main(["--local-dir", "/data/DL3DV-ALL-960P", "--scene-id", "scene_a", "--subdir", "8K"])

        self.assertEqual(exit_code, 0)
        download.assert_called_once_with(
            repo_id=DL3DV_DATASET,
            repo_type="dataset",
            filename="8K/scene_a.zip",
            local_dir="/data/DL3DV-ALL-960P",
        )

    def test_requires_explicit_local_dir_scene_id_and_subdir(self) -> None:
        for argv in (
            ["--scene-id", SCENE_ID, "--subdir", SUBDIR],
            ["--local-dir", "/data/DL3DV-ALL-960P", "--subdir", SUBDIR],
            ["--local-dir", "/data/DL3DV-ALL-960P", "--scene-id", SCENE_ID],
        ):
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as cm:
                        main(argv)
                self.assertNotEqual(cm.exception.code, 0)

if __name__ == "__main__":
    unittest.main()
