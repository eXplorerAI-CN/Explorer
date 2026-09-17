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

"""Render an arbitrary camera trajectory from a JSON file of 3x4 or 4x4 poses.

General-purpose script for rendering any pre-computed trajectory through a
3dgrut checkpoint. Supports optional dataset image downsampling via
--downsample for resolution control.

Usage:
    python render_trajectory.py \\
        --checkpoint /path/to/ckpt_30000.pt \\
        --out-dir /path/to/output \\
        --trajectory-file /path/to/trajectory.json \\
        [--downsample 2]
"""

import argparse
from pathlib import Path

from threedgrut.render import Renderer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="path to the pretrained checkpoint")
    parser.add_argument(
        "--path", type=str, default="", help="Path to the training data, if not provided taken from ckpt"
    )
    parser.add_argument("--out-dir", required=True, type=str, help="Output path")
    parser.add_argument("--trajectory-file", required=True, type=Path, help="Path to the trajectory file")
    parser.add_argument("--downsample", type=int, default=None, help="Downsample factor for dataset images")
    args = parser.parse_args()

    config_overrides = {}
    if args.downsample is not None:
        config_overrides["dataset.downsample_factor"] = args.downsample

    renderer = Renderer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        path=args.path,
        out_dir=args.out_dir,
        save_gt=False,
        computes_extra_metrics=False,
        config_overrides=config_overrides if config_overrides else None,
    )

    renderer.render_from_file(args.trajectory_file)