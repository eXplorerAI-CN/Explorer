# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
from typing import Any, BinaryIO

import av
import numpy as np

FrameSource = Any


def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if np.issubdtype(frame.dtype, np.floating):
        frame = (np.clip(frame, 0, 1) * 255).round().astype(np.uint8)
    elif frame.dtype != np.uint8:
        raise TypeError(f"Expected uint8 or float video frames, got {frame.dtype}")
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got shape {frame.shape}")
    return np.ascontiguousarray(frame)


def _is_torch_tensor(value: object) -> bool:
    if type(value).__module__.split(".", 1)[0] != "torch":
        return False
    import torch

    return torch.is_tensor(value)


def _iter_frame_arrays(frames: FrameSource):
    if _is_torch_tensor(frames):
        if frames.ndim != 4:
            raise ValueError(f"Expected TCHW tensor, got shape {tuple(frames.shape)}")
        for frame in frames:
            yield _as_uint8_rgb(frame.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy())
        return

    if isinstance(frames, np.ndarray):
        for frame in frames:
            yield _as_uint8_rgb(frame)
        return

    from PIL import Image

    for path in frames:
        with Image.open(path) as image:
            yield _as_uint8_rgb(np.asarray(image.convert("RGB")))


def _write_video(
    frames: FrameSource,
    output: str | Path | BinaryIO,
    fps: int,
    codec: str,
    crf: int,
    pix_fmt: str,
) -> None:
    frame_iter = iter(_iter_frame_arrays(frames))
    try:
        first = next(frame_iter)
    except StopIteration as exc:
        raise ValueError("Cannot write a video with zero frames") from exc

    height, width = first.shape[:2]
    if pix_fmt == "yuv420p" and (width % 2 or height % 2):
        pix_fmt = "yuv444p"

    if isinstance(output, Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        output = str(output)

    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = pix_fmt
        stream.options = {"crf": str(crf)}

        def encode(frame_array: np.ndarray) -> None:
            frame_array = _as_uint8_rgb(frame_array)
            if frame_array.shape[:2] != (height, width):
                raise ValueError(f"Video frame size changed from {(height, width)} to {frame_array.shape[:2]}")
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24").reformat(format=pix_fmt)
            for packet in stream.encode(frame):
                container.mux(packet)

        encode(first)
        for frame_array in frame_iter:
            encode(frame_array)
        for packet in stream.encode():
            container.mux(packet)


def save_video(frames: FrameSource, path: Path, fps: int = 15, crf: int = 23) -> None:
    _write_video(frames, path, fps=fps, codec="libx264", crf=crf, pix_fmt="yuv420p")


def encode_frames_to_video(
    frames: np.ndarray, codec: str = "libx264", crf: int = 23, pix_fmt: str = "yuv420p", fps: int = 30
) -> bytes:
    output = io.BytesIO()
    _write_video(frames, output, fps=fps, codec=codec, crf=crf, pix_fmt=pix_fmt)
    return output.getvalue()


def decode_video_to_frames(video_bytes: bytes) -> np.ndarray:
    with av.open(io.BytesIO(video_bytes)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def decode_video_frame_count(video_bytes: bytes) -> int:
    frame_count = 0
    with av.open(io.BytesIO(video_bytes)) as container:
        for frame in container.decode(video=0):
            frame.to_ndarray(format="rgb24")
            frame_count += 1
    return frame_count
