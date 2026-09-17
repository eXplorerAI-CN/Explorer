# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert frame-based HDF5 files to the video-compressed HDF5 format used by training.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from model_training.utils.video_io import decode_video_to_frames, encode_frames_to_video


def encode_depth_frames(depth_frames: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Quantize float depth frames to float16 for compact HDF5 storage."""
    return float(depth_frames.min()), float(depth_frames.max()), depth_frames.astype(np.float16)


def decode_depth_frames(depth_frames: np.ndarray) -> np.ndarray:
    """Decode quantized depth frames."""
    return depth_frames.astype(np.float32)


def convert_hdf5_to_video(
    input_path: Path, output_path: Path, crf: int = 23, codec: str = "libx264", quantize_depth: bool = True
) -> None:
    """
    Convert a frame-based HDF5 file to video-compressed HDF5.

    The input file must contain render_np_array, opacity_np_array, depth_np_array,
    selected_indices, and scale datasets.
    """
    print(f"Converting {input_path} -> {output_path}")

    with h5py.File(input_path, "r") as hf_in:
        render_frames = hf_in["render_np_array"][:]
        opacity_frames = hf_in["opacity_np_array"][:]
        depth_frames = hf_in["depth_np_array"][:]
        selected_indices = hf_in["selected_indices"][:]
        scale = hf_in["scale"][()]

    n_frames = render_frames.shape[0]
    print(f"  Frames: {n_frames}")
    print(f"  Render shape: {render_frames.shape}")
    print(f"  Opacity shape: {opacity_frames.shape}")
    print(f"  Depth shape: {depth_frames.shape}")

    print("  Encoding renders...")
    render_video = encode_frames_to_video(render_frames, codec=codec, crf=crf)
    print(f"    Render video size: {len(render_video) / 1024 / 1024:.2f} MB")

    print("  Encoding opacity...")
    opacity_rgb = np.repeat(opacity_frames, 3, axis=-1)
    opacity_video = encode_frames_to_video(opacity_rgb, codec=codec, crf=crf)
    print(f"    Opacity video size: {len(opacity_video) / 1024 / 1024:.2f} MB")

    depth_min = depth_max = None
    depth_quantized = None
    if quantize_depth:
        print("  Quantizing depth...")
        depth_min, depth_max, depth_quantized = encode_depth_frames(depth_frames)
        print(f"    Depth range: [{depth_min:.4f}, {depth_max:.4f}]")

    with h5py.File(output_path, "w") as hf_out:
        hf_out.attrs["format"] = "video_compressed"
        hf_out.attrs["codec"] = codec
        hf_out.attrs["crf"] = crf
        hf_out.attrs["n_frames"] = n_frames
        hf_out.attrs["render_shape"] = render_frames.shape[1:]
        hf_out.attrs["opacity_shape"] = opacity_frames.shape[1:]
        hf_out.attrs["depth_shape"] = depth_frames.shape[1:]

        hf_out.create_dataset("render_video", data=np.frombuffer(render_video, dtype=np.uint8))
        hf_out.create_dataset("opacity_video", data=np.frombuffer(opacity_video, dtype=np.uint8))
        hf_out.create_dataset("selected_indices", data=selected_indices)
        hf_out.create_dataset("scale", data=scale)

        if quantize_depth:
            hf_out.create_dataset("depth_min", data=depth_min)
            hf_out.create_dataset("depth_max", data=depth_max)
            hf_out.create_dataset("depth_quantized", data=depth_quantized, compression="gzip", compression_opts=4)
        else:
            hf_out.create_dataset("depth_np_array", data=depth_frames, compression="gzip", compression_opts=4)

    input_size = input_path.stat().st_size / 1024 / 1024
    output_size = output_path.stat().st_size / 1024 / 1024
    print(f"  Input size:  {input_size:.2f} MB")
    print(f"  Output size: {output_size:.2f} MB")
    print(f"  Compression ratio: {input_size / output_size:.2f}x")


class VideoHDF5Dataset:
    """Data loader for video-compressed HDF5 files."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.hf = h5py.File(self.path, "r")
        self.n_frames = self.hf.attrs["n_frames"]
        self._renders = None
        self._opacity = None
        self._depth = None

    def _load_renders(self) -> np.ndarray:
        if self._renders is None:
            self._renders = decode_video_to_frames(self.hf["render_video"][:].tobytes())
        return self._renders

    def _load_opacity(self) -> np.ndarray:
        if self._opacity is None:
            opacity_rgb = decode_video_to_frames(self.hf["opacity_video"][:].tobytes())
            self._opacity = opacity_rgb[..., :1]
        return self._opacity

    def _load_depth(self) -> np.ndarray:
        if self._depth is None:
            if "depth_quantized" in self.hf:
                self._depth = decode_depth_frames(self.hf["depth_quantized"][:])
            else:
                self._depth = self.hf["depth_np_array"][:]
        return self._depth

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {
            "render": self._load_renders()[idx],
            "opacity": self._load_opacity()[idx],
            "depth": self._load_depth()[idx],
        }

    def get_all(self) -> dict[str, np.ndarray | float]:
        return {
            "render": self._load_renders(),
            "opacity": self._load_opacity(),
            "depth": self._load_depth(),
            "selected_indices": self.hf["selected_indices"][:],
            "scale": float(self.hf["scale"][()]),
        }

    def close(self) -> None:
        self.hf.close()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert frame-based HDF5 to video-compressed HDF5")
    subparsers = parser.add_subparsers(dest="command", required=True)

    video_parser = subparsers.add_parser("video", help="Convert to video-compressed HDF5")
    video_parser.add_argument("--input", "-i", type=Path, required=True, help="Input HDF5 file")
    video_parser.add_argument("--output", "-o", type=Path, help="Output HDF5 file")
    video_parser.add_argument(
        "--crf", type=int, default=23, help="Constant Rate Factor. Lower values give higher quality."
    )
    video_parser.add_argument("--codec", type=str, default="libx264", choices=["libx264", "libx265"])
    video_parser.add_argument("--no-quantize-depth", action="store_true", help="Store depth as float32 instead")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output is not None else args.input.with_suffix(".video.h5")
    convert_hdf5_to_video(
        input_path=args.input,
        output_path=output_path,
        crf=args.crf,
        codec=args.codec,
        quantize_depth=not args.no_quantize_depth,
    )


if __name__ == "__main__":
    main()
