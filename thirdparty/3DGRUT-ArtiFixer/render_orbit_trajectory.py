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

"""Render an orbit trajectory built from actual training/test camera poses.

Sorts all camera poses by their angle around the estimated scene center
(using PCA), then interpolates between consecutive poses with SLERP.
Requires a scale file from metric alignment to convert interp_distance
from real-world units to the scene's coordinate frame.

Used for generating smooth orbit videos that pass through actual camera
viewpoints, e.g. for qualitative evaluation or visualization.

Usage:
    python render_orbit_trajectory.py \\
        --checkpoint /path/to/ckpt_30000.pt \\
        --out-dir /path/to/output \\
        --scale-file /path/to/scale_info.txt \\
        --training-pose-start 0 \\
        [--interp-distance 0.1]
"""

import argparse
from pathlib import Path

from threedgrut.render import Renderer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render orbit trajectory from actual camera poses")
    parser.add_argument("--checkpoint", required=True, type=str, help="path to the pretrained checkpoint")
    parser.add_argument(
        "--path", type=str, default="", help="Path to the training data, if not provided taken from ckpt"
    )
    parser.add_argument("--out-dir", required=True, type=str, help="Output path")
    parser.add_argument("--scale-file", required=True, type=Path, help="Path to the scale info file")
    parser.add_argument("--training-pose-start", type=int, required=True, help="Index of the first training pose to include in the orbit trajectory")
    parser.add_argument("--interp-distance", type=float, default=0.1, help="Distance between consecutive frames on the orbit")
    args = parser.parse_args()

    renderer = Renderer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        path=args.path,
        out_dir=args.out_dir,
        save_gt=False,
        computes_extra_metrics=False,
    )

    with args.scale_file.open() as f:
        scale = float(f.read().split("Scale factor: ")[1])

    print(f"Scale factor: {scale}")
    renderer.render_orbit_trajectory(args.interp_distance, scale, args.training_pose_start)
