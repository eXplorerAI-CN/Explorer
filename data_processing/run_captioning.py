# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate ArtiFixer prompt HDF5 files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from data_processing.captioning.caption_plan import caption_output_path, caption_variants
from data_processing.captioning.generate_captions import generate_caption_hdf5
from data_processing.scene_utils import (
    discover_scene_zips,
    read_scene_list,
    resolve_scene_path,
    scene_name,
    scene_relative_parent,
)


def generate_for_scene(args: argparse.Namespace, scene_path: Path) -> None:
    assert scene_path.suffix == ".zip", f"DL3DV captioning expects scene zip inputs: {scene_path}"
    scene_output_dir = args.output_dir / scene_relative_parent(args.dl3dv_dir, scene_path) / scene_name(scene_path)
    if args.hf_cache_dir is not None:
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(args.hf_cache_dir)
    if args.hf_hub_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    for frame_stride, revisit, reverse in caption_variants(
        args.max_frame_stride,
        include_reverse=args.include_reverse,
        include_revisit=args.include_revisit,
    ):
        output_path = caption_output_path(
            scene_output_dir, args.num_frames, frame_stride, revisit=revisit, reverse=reverse
        )
        generate_caption_hdf5(
            scene_path,
            output_path,
            num_frames=args.num_frames,
            frame_stride=frame_stride,
            revisit=revisit,
            reverse=reverse,
            captioning_model_id=args.captioning_model_id,
            captioning_attn_implementation=args.captioning_attn_implementation,
            text_encoder_model_id=args.text_encoder_model_id,
            check_if_exists=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dl3dv_dir", type=Path, required=True, help="Root containing DL3DV scene zips")
    parser.add_argument("--output_dir", type=Path, required=True, help="Root for generated prompt HDF5 files")
    parser.add_argument("--scene_id", action="append", default=[], help="Only process this scene id/path")
    parser.add_argument("--scene_list", type=Path, help="Only process scenes listed in this file")
    parser.add_argument("--num_frames", type=int, default=81, help="Prompt window length; <=0 means all frames")
    parser.add_argument("--max_frame_stride", type=int, default=1)
    parser.add_argument("--include_reverse", action="store_true", help="Also generate *_reverse prompt files")
    parser.add_argument("--include_revisit", action="store_true", help="Also generate *_revisit prompt files")
    parser.add_argument("--hf_cache_dir", type=Path)
    parser.add_argument("--hf_hub_offline", action="store_true", help="Use only already-cached Hugging Face assets")
    parser.add_argument("--captioning_model_id", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--captioning_attn_implementation")
    parser.add_argument("--text_encoder_model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    args = parser.parse_args()

    if args.max_frame_stride < 1:
        parser.error("--max_frame_stride must be >= 1")

    scenes = list(args.scene_id)
    if args.scene_list is not None:
        scenes.extend(read_scene_list(args.scene_list))
    if scenes:
        args.scene_paths = [resolve_scene_path(args.dl3dv_dir, scene) for scene in scenes]
    else:
        args.scene_paths = discover_scene_zips(args.dl3dv_dir)
        if not args.scene_paths:
            parser.error(f"No scene zips found under {args.dl3dv_dir}")
    return args


def main() -> None:
    args = parse_args()
    for scene_path in args.scene_paths:
        generate_for_scene(args, scene_path)


if __name__ == "__main__":
    main()
