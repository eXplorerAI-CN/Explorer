# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import unittest


class Artifixer3DDistillationTests(unittest.TestCase):
    def test_release_distillation_defaults_to_scratch30k(self):
        run_source = Path("data_processing/run_artifixer3d.py").read_text()
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn("--artifixer3d_steps", run_source)
        self.assertIn("default=30000", run_source)
        self.assertNotIn("resume_mode", run_source + artifixer3d_source)
        self.assertNotIn("image_path_override_fallback_to_original", run_source + artifixer3d_source)

    def test_resume_requires_explicit_base_checkpoint(self):
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn("base_checkpoint: Path | None", artifixer3d_source)
        self.assertIn("if base_checkpoint is not None:", artifixer3d_source)
        self.assertIn('overrides.append(f"resume={base_checkpoint}")', artifixer3d_source)

    def test_distillation_passes_selected_indices_and_image_override(self):
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn('f"selected_indices_file={paths.distillation_selected_indices_path}"', artifixer3d_source)
        self.assertIn('f"image_path_override={paths.override_image_dir.name}"', artifixer3d_source)
        self.assertIn('f"n_iterations={steps}"', artifixer3d_source)


if __name__ == "__main__":
    unittest.main()
