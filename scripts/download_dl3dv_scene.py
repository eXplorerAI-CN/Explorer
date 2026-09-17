#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

DL3DV_DATASET = "DL3DV/DL3DV-ALL-960P"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one DL3DV scene archive from Hugging Face.")
    parser.add_argument("--local-dir", type=Path, required=True, help="Destination DL3DV root")
    parser.add_argument("--scene-id", required=True, help="DL3DV scene id to download")
    parser.add_argument("--subdir", required=True, help="DL3DV dataset subdirectory containing the scene zip")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    filename = f"{args.subdir}/{args.scene_id}.zip"
    path = hf_hub_download(
        repo_id=DL3DV_DATASET,
        repo_type="dataset",
        filename=filename,
        local_dir=str(args.local_dir),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
