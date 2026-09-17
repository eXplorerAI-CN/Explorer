# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for DL3DV datasets.

This module provides common functionality for:
- Inference pair generation (NeighborSelectionMode, InferencePair, generate_inference_pairs)
- Prompt loading
- Camera ray computation
- Image loading
"""

import warnings
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, List, Set, Tuple
from zipfile import ZipFile

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation as R
from threedgrut.datasets.camera_models import OpenCVPinholeCameraModelParameters, ShutterType, pixels_to_image_points

from model_training.constants import MAX_SEQUENCE_LENGTH
from model_training.data.scene_utils import downsampled_image_path, scene_zip_member
from model_training.utils.pose_utils import invert_SE3

# =============================================================================
# Inference Pair Generation
# =============================================================================


class NeighborSelectionMode(Enum):
    """Mode for selecting neighbor indices for inference."""

    CONSECUTIVE = "consecutive"
    EVENLY_SPACED = "evenly_spaced"
    COVISIBILITY = "covisibility"


@dataclass
class InferencePair:
    """A pair of neighbor/test indices for model inference."""

    neighbor_indices: List[int]  # Neighbor context indices
    test_indices: List[int]  # Frames to run inference on (may include train frames for constant fps)
    reversed: bool
    chunk_idx: int
    scene_id: str = ""
    is_test_frame: List[bool] = None  # True for actual test frames, False for train frames included for constant fps


def so3_relative_angle(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Compute the angle between rotation matrices."""
    prod = np.matmul(np.swapaxes(R1, -2, -1), R2)
    cos = (np.trace(prod, axis1=-2, axis2=-1) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    return np.abs(np.arccos(cos))


def pose_distance_matrix(extrinsics_c2w: np.ndarray, lambda_t: float = 1.0) -> np.ndarray:
    """Calculate pairwise pose distances between all cameras."""
    rs = extrinsics_c2w[:, :3, :3]
    ts = extrinsics_c2w[:, :3, 3]
    avg_scale = np.mean(np.linalg.norm(ts, axis=1))
    if avg_scale > 0:
        ts = ts / avg_scale

    t_dists = np.linalg.norm(ts[:, None, :] - ts[None, :, :], axis=-1)
    r_dists = so3_relative_angle(rs[:, None, ...], rs[None, ...])
    r_dists = np.rad2deg(r_dists) / 180.0
    return r_dists + lambda_t * t_dists


def select_neighbor_indices_consecutive(
    train_ids: Set[int], test_indices: List[int], num_train: int = 12
) -> Tuple[List[int], bool]:
    """Select closest block of consecutive train indices."""
    sorted_train = sorted(train_ids)
    test_start, test_end = min(test_indices), max(test_indices)

    blocks = []
    current_block = [sorted_train[0]]
    for i in range(1, len(sorted_train)):
        if sorted_train[i] == sorted_train[i - 1] + 1:
            current_block.append(sorted_train[i])
        else:
            if len(current_block) >= num_train:
                blocks.append(current_block)
            current_block = [sorted_train[i]]
    if len(current_block) >= num_train:
        blocks.append(current_block)

    if not blocks:
        return [], False

    best_indices = None
    best_dist = float("inf")
    use_reversed = False

    for block in blocks:
        if block[-1] < test_start:
            indices = block[-num_train:]
            dist = test_start - block[-1]
            if dist < best_dist:
                best_dist = dist
                best_indices = indices
                use_reversed = False
        elif block[0] > test_end:
            indices = block[:num_train]
            dist = block[0] - test_end
            if dist < best_dist:
                best_dist = dist
                best_indices = indices
                use_reversed = True

    return best_indices if best_indices else [], use_reversed


def select_neighbor_indices_evenly_spaced(
    train_ids: Set[int], test_indices: List[int], num_train: int = 12
) -> Tuple[List[int], bool]:
    """Select evenly spaced indices from available train frames."""
    sorted_train = sorted(train_ids)
    if len(sorted_train) < num_train:
        return [], False

    step = len(sorted_train) / num_train
    indices = [sorted_train[int(i * step)] for i in range(num_train)]

    test_start = min(test_indices)
    test_end = max(test_indices)
    train_center = (indices[0] + indices[-1]) / 2
    test_center = (test_start + test_end) / 2

    use_reversed = train_center > test_center

    return indices, use_reversed


def select_neighbor_indices_covisibility(
    train_ids: Set[int], test_indices: List[int], extrinsics_c2w: np.ndarray, num_train: int = 12
) -> Tuple[List[int], bool]:
    """Select train views that maximize covisibility with test views."""
    sorted_train = sorted(train_ids)
    if len(sorted_train) < num_train:
        return [], False

    dist_matrix = pose_distance_matrix(extrinsics_c2w)

    train_to_test_dists = {}
    for t_idx in sorted_train:
        avg_dist = np.mean([dist_matrix[t_idx, test_idx] for test_idx in test_indices])
        train_to_test_dists[t_idx] = avg_dist

    selected = sorted(train_to_test_dists.keys(), key=lambda x: train_to_test_dists[x])[:num_train]
    selected = sorted(selected)

    test_start = min(test_indices)
    test_end = max(test_indices)
    train_center = (selected[0] + selected[-1]) / 2
    test_center = (test_start + test_end) / 2

    use_reversed = train_center > test_center

    return selected, use_reversed


def generate_inference_pairs(
    train_ids: Set[int],
    total_frames: int,
    num_train_context: int = 12,
    max_test_frames: int | None = None,
    selection_mode: NeighborSelectionMode = NeighborSelectionMode.COVISIBILITY,
    extrinsics_c2w: np.ndarray = None,
    test_ids: Set[int] | None = None,
    include_all_frames: bool = False,
) -> List[InferencePair]:
    """Generate (neighbor_indices, test_indices) pairs to cover the requested eval trajectory.

    Args:
        train_ids: Set of frame indices used for reconstruction/training.
        total_frames: Total number of frames in the scene.
        num_train_context: Number of training frames to use as context.
        max_test_frames: Maximum eval frames per chunk. None means one chunk per contiguous segment.
        selection_mode: How to select training context frames.
        extrinsics_c2w: Camera poses, required for covisibility selection.
        test_ids: Optional explicit set of test frame indices. If None, test frames are the complement of train_ids.
        include_all_frames: If True, cover the full source trajectory instead of only held-out/test frames.
            Explicit target trajectories and full-clip rendering always preserve frame order.
    """
    assert (
        max_test_frames is None or max_test_frames > 0
    ), f"max_test_frames must be positive when set, got {max_test_frames}"

    if test_ids is not None:
        is_test = np.array([i in test_ids for i in range(total_frames)])
    else:
        is_test = np.array([i not in train_ids for i in range(total_frames)])
    preserve_frame_order = include_all_frames or test_ids is not None

    def select_neighbors(target_indices: List[int]) -> Tuple[List[int], bool]:
        if selection_mode == NeighborSelectionMode.CONSECUTIVE:
            return select_neighbor_indices_consecutive(train_ids, target_indices, num_train_context)
        if selection_mode == NeighborSelectionMode.EVENLY_SPACED:
            return select_neighbor_indices_evenly_spaced(train_ids, target_indices, num_train_context)
        assert extrinsics_c2w is not None, "extrinsics_c2w required for COVISIBILITY mode"
        return select_neighbor_indices_covisibility(train_ids, target_indices, extrinsics_c2w, num_train_context)

    inference_pairs: List[InferencePair] = []

    def append_pair(indices: List[int], is_test_frame: List[bool], preserve_order: bool) -> None:
        target_indices = [idx for idx, keep in zip(indices, is_test_frame) if keep] or indices
        neighbor_indices, use_reversed = select_neighbors(target_indices)
        if not neighbor_indices:
            print(f"Warning: Could not find neighbor indices for eval chunk [{indices[0]}, {indices[-1]}]")
            return

        if not preserve_order and use_reversed:
            indices = indices[::-1]
            is_test_frame = is_test_frame[::-1]
            neighbor_indices = neighbor_indices[::-1]

        inference_pairs.append(
            InferencePair(
                neighbor_indices=neighbor_indices,
                test_indices=indices,
                reversed=not preserve_order and use_reversed,
                chunk_idx=len(inference_pairs),
                is_test_frame=is_test_frame,
            )
        )

    if include_all_frames:
        all_indices = list(range(total_frames))
        chunk_size = len(all_indices) if max_test_frames is None else max_test_frames
        for chunk_start in range(0, len(all_indices), chunk_size):
            chunk = all_indices[chunk_start : chunk_start + chunk_size]
            append_pair(chunk, [bool(is_test[idx]) for idx in chunk], preserve_order=True)
        return inference_pairs

    test_segments = []
    in_test_segment = False
    segment_start = 0

    for i in range(total_frames):
        if is_test[i] and not in_test_segment:
            in_test_segment = True
            segment_start = i
        elif not is_test[i] and in_test_segment:
            in_test_segment = False
            test_segments.append((segment_start, i - 1))

    if in_test_segment:
        test_segments.append((segment_start, total_frames - 1))

    for seg_start, seg_end in test_segments:
        remaining_indices = list(range(seg_start, seg_end + 1))
        while remaining_indices:
            chunk_size = (
                len(remaining_indices) if max_test_frames is None else min(len(remaining_indices), max_test_frames)
            )
            test_chunk = remaining_indices[:chunk_size]
            remaining_indices = remaining_indices[chunk_size:]
            append_pair(test_chunk, [True] * len(test_chunk), preserve_order=preserve_frame_order)

    return inference_pairs


# =============================================================================
# Prompt Loading
# =============================================================================


def load_encoded_prompt(
    prompt_paths: List[Path],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, str]:
    """
    Load an encoded prompt from a random prompt file.

    Args:
        prompt_paths: List of paths to prompt HDF5 files
        generator: Random generator for selection

    Returns:
        Tuple of (encoded_prompt tensor, caption string)
    """
    if not prompt_paths:
        return (torch.zeros(MAX_SEQUENCE_LENGTH, 4096, dtype=torch.bfloat16), "")

    prompt_idx = torch.randint(0, len(prompt_paths), size=(1,), generator=generator).item()
    prompt_path = prompt_paths[prompt_idx]

    with h5py.File(prompt_path, "r") as f:
        start_frame_candidates = list(f.keys())
        prompt_dataset = f[start_frame_candidates[0]]

        encoded_prompt = torch.tensor(prompt_dataset[:], dtype=torch.uint16).view(torch.bfloat16)
        encoded_prompt = torch.cat(
            [
                encoded_prompt,
                torch.zeros(
                    MAX_SEQUENCE_LENGTH - encoded_prompt.shape[0],
                    encoded_prompt.shape[1],
                    dtype=torch.bfloat16,
                ),
            ],
            dim=0,
        )

        caption = prompt_dataset.attrs.get("caption", "")

    return encoded_prompt, caption


def load_frames_from_paths(
    base_dir: Path,
    scene_id: str,
    file_paths: List[str],
    downsample_factor: int = 4,
) -> torch.Tensor:
    """
    Load frames from disk given file paths.

    Args:
        base_dir: Base directory containing scene folders
        scene_id: Scene identifier
        file_paths: List of relative file paths within the scene
        downsample_factor: Downsample factor for image path substitution

    Returns:
        Tensor of shape (N, C, H, W) with values in [0, 1]
    """
    frames = []
    for file_path in file_paths:
        file_path = downsampled_image_path(file_path, downsample_factor)
        with Image.open(base_dir / scene_id / file_path) as image:
            frames.append(np.asarray(image.convert("RGB")))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.FloatTensor(np.stack(frames)).permute(0, 3, 1, 2) / 255.0


def load_frames_from_zip(
    zf: ZipFile,
    scene_id: str,
    file_paths: List[str],
    scene_has_subdir: bool,
    downsample_factor: int = 4,
) -> torch.Tensor:
    """Load frames from a zip file."""
    frames = []
    for file_path in file_paths:
        file_path = downsampled_image_path(file_path, downsample_factor)
        with zf.open(scene_zip_member(scene_id if scene_has_subdir else "", file_path), "r") as f:
            with Image.open(BytesIO(f.read())) as image:
                frames.append(np.asarray(image.convert("RGB")))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.FloatTensor(np.stack(frames)).permute(0, 3, 1, 2) / 255.0



def load_frames_from_dir(scene_root: Path, file_paths: List[str], downsample_factor: int = 4) -> torch.Tensor:
    """Load frames from an extracted DL3DV scene directory."""
    frames = []
    for file_path in file_paths:
        image_path = scene_root / downsampled_image_path(file_path, downsample_factor)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing frame: {image_path}")
        with Image.open(image_path) as image:
            frames.append(np.asarray(image.convert("RGB")))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.FloatTensor(np.stack(frames)).permute(0, 3, 1, 2) / 255.0


def load_indexed_frames(
    render_dir: Path,
    frame_indices: List[int],
    filename_format: str = "{:04d}.png",
    grayscale: bool = False,
) -> torch.Tensor:
    """
    Load frames by index from a directory.

    Args:
        render_dir: Directory containing numbered PNG files
        frame_indices: List of frame indices to load
        filename_format: Format string for filenames (default: 0000.png)
        grayscale: Whether to load as grayscale

    Returns:
        Tensor of shape (N, C, H, W) with values in [0, 1]
    """
    frames = []
    for idx in frame_indices:
        img_path = render_dir / filename_format.format(idx)
        if not img_path.exists():
            raise FileNotFoundError(f"Missing frame: {img_path}")
        image = Image.open(img_path)
        if grayscale:
            image = image.convert("L")
        frames.append(np.asarray(image))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if grayscale:
            return torch.FloatTensor(np.stack(frames)).unsqueeze(1) / 255.0
        return torch.FloatTensor(np.stack(frames)).permute(0, 3, 1, 2) / 255.0


def resize_to_multiple_of_16(tensor: torch.Tensor) -> torch.Tensor:
    """Resize tensor spatial dimensions to be multiples of 16."""
    h, w = tensor.shape[-2:]
    new_h = round(h / 16) * 16
    new_w = round(w / 16) * 16
    if new_h != h or new_w != w:
        return F.interpolate(tensor, (new_h, new_w), mode="bilinear")
    return tensor


_INTRINSIC_KEYS = ("w", "h", "fl_x", "fl_y", "cx", "cy")
_DISTORTION_KEYS = ("k1", "k2", "p1", "p2")


def _intrinsics_vector(intrinsics: dict) -> torch.Tensor:
    for key in _INTRINSIC_KEYS:
        assert key in intrinsics, f"Camera intrinsics missing {key!r}"
    w = float(intrinsics["w"])
    h = float(intrinsics["h"])
    assert w > 0 and h > 0, f"Camera image size must be positive, got {(w, h)}"
    return torch.FloatTensor(
        [
            float(intrinsics["fl_x"]) / w,
            float(intrinsics["fl_y"]) / h,
            float(intrinsics["cx"]) / w,
            float(intrinsics["cy"]) / h,
            float(intrinsics.get("k1", 0.0)),
            float(intrinsics.get("k2", 0.0)),
            float(intrinsics.get("p1", 0.0)),
            float(intrinsics.get("p2", 0.0)),
        ]
    )


def _frame_intrinsics(transforms: dict, frame: dict) -> torch.Tensor:
    intrinsics = dict(transforms)
    for key in (*_INTRINSIC_KEYS, *_DISTORTION_KEYS):
        if key in frame:
            intrinsics[key] = frame[key]
    return _intrinsics_vector(intrinsics)


def _intrinsics_tensor(intrinsics: dict, count: int) -> torch.Tensor:
    return _intrinsics_vector(intrinsics).unsqueeze(0).expand(count, -1).clone()


def _average_temporal_intrinsics(intrinsics: torch.Tensor, skip_vae_check: bool) -> torch.Tensor:
    averaged = [intrinsics[0]]
    num_frames = intrinsics.shape[0]
    for i in range(1, num_frames, 4):
        group = intrinsics[i : min(i + 4, num_frames)]
        if skip_vae_check and group.shape[0] < 4:
            group = torch.cat([group, group[-1:].expand(4 - group.shape[0], -1)], dim=0)
        averaged.append(group.mean(dim=0))
    return torch.stack(averaged)


def _intrinsics_to_K(intrinsics: torch.Tensor) -> torch.Tensor:
    K = torch.zeros(intrinsics.shape[0], 3, 3, dtype=intrinsics.dtype, device=intrinsics.device)
    K[:, 0, 0] = intrinsics[:, 0]
    K[:, 1, 1] = intrinsics[:, 1]
    K[:, 0, 2] = intrinsics[:, 2] - 0.5
    K[:, 1, 2] = intrinsics[:, 3] - 0.5
    K[:, 2, 2] = 1.0
    return K


def _camera_rays_for_intrinsics(
    intrinsics: torch.Tensor,
    averaged_c2ws_ref: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    u = np.tile(np.arange(image_width), image_height)
    v = np.arange(image_height).repeat(image_width)
    pixel_coords = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32)
    image_points = pixels_to_image_points(pixel_coords)

    rays_d_cam = []
    for fx_norm, fy_norm, cx_norm, cy_norm, k1, k2, p1, p2 in intrinsics.detach().cpu().tolist():
        params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([image_width, image_height]).astype(np.int64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([cx_norm * image_width, cy_norm * image_height]).astype(np.float32),
            focal_length=np.array([fx_norm * image_width, fy_norm * image_height]).astype(np.float32),
            radial_coeffs=np.array([k1, k2, 0, 0, 0, 0]).astype(np.float32),
            tangential_coeffs=np.array([p1, p2]).astype(np.float32),
            thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
        )
        rays_d_cam.append(torch.as_tensor(params._image_points_to_camera_rays_impl(image_points), dtype=torch.float32))
    rays_d_cam = torch.stack(rays_d_cam)
    rays_d_cam = rays_d_cam.to(device=averaged_c2ws_ref.device, dtype=averaged_c2ws_ref.dtype)
    rays_d_ref = torch.bmm(rays_d_cam, averaged_c2ws_ref[:, :3, :3].transpose(1, 2))
    rays_o_ref = averaged_c2ws_ref[:, :3, 3].unsqueeze(1).expand(-1, rays_d_ref.shape[1], -1)
    rays_dxo_ref = torch.linalg.cross(rays_o_ref, rays_d_ref)
    return torch.cat([rays_dxo_ref, rays_d_ref], dim=-1).view(rays_d_ref.shape[0], image_height, image_width, 6)


def compute_camera_conditioning(
    intrinsics: dict,
    c2ws_world: torch.Tensor,
    neighbor_c2ws_world: torch.Tensor,
    scale: float,
    image_height: int,
    image_width: int,
    skip_vae_check: bool = False,
    target_intrinsics: torch.Tensor | None = None,
    neighbor_intrinsics: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute camera conditioning (prope w2cs/Ks, neighbor w2cs/Ks, plucker rays).

    Args:
        intrinsics: Default camera intrinsics used when per-frame intrinsics are not provided.
        c2ws_world: (N, 4, 4) camera-to-world transforms for main frames.
        neighbor_c2ws_world: (M, 4, 4) camera-to-world transforms for neighbor frames.
        scale: Scale factor for translations.
        image_height: Height of the images.
        image_width: Width of the images.
        skip_vae_check: If True, allow frame counts that don't satisfy 1 + 4*n.
        target_intrinsics: Optional (N, 8) normalized per-frame intrinsics.
        neighbor_intrinsics: Optional (M, 8) normalized per-neighbor intrinsics.

    Returns:
        Dict with w2cs, Ks, neighbor_w2cs, neighbor_Ks, camera_rays.
    """
    item = {}

    c2ws_world = c2ws_world.clone()
    device = c2ws_world.device
    neighbor_c2ws_world = neighbor_c2ws_world.clone().to(device=device, dtype=c2ws_world.dtype)
    c2ws_world[:, :3, 3] = c2ws_world[:, :3, 3] * scale
    if target_intrinsics is None:
        target_intrinsics = _intrinsics_tensor(intrinsics, c2ws_world.shape[0])
    else:
        assert target_intrinsics.shape == (
            c2ws_world.shape[0],
            8,
        ), f"target_intrinsics must have shape {(c2ws_world.shape[0], 8)}, got {tuple(target_intrinsics.shape)}"
    if neighbor_intrinsics is None:
        neighbor_intrinsics = _intrinsics_tensor(intrinsics, neighbor_c2ws_world.shape[0])
    else:
        assert neighbor_intrinsics.shape == (neighbor_c2ws_world.shape[0], 8), (
            f"neighbor_intrinsics must have shape {(neighbor_c2ws_world.shape[0], 8)}, got "
            f"{tuple(neighbor_intrinsics.shape)}"
        )

    target_intrinsics = target_intrinsics.to(device=device, dtype=c2ws_world.dtype)
    neighbor_intrinsics = neighbor_intrinsics.to(device=device, dtype=c2ws_world.dtype)
    ref_w2c = invert_SE3(c2ws_world[0])
    c2ws_ref = ref_w2c @ c2ws_world

    # Average camera poses in groups of 4 for VAE temporal compression.
    averaged_c2ws_ref = [c2ws_ref[0]]
    if not skip_vae_check:
        assert (c2ws_ref.shape[0] - 1) % 4 == 0, f"Frames should be 1 + 4*n for VAE, got {c2ws_ref.shape[0]}"
    num_frames = c2ws_ref.shape[0]
    for i in range(1, num_frames, 4):
        end_idx = min(i + 4, num_frames)
        group = c2ws_ref[i:end_idx]
        if skip_vae_check and group.shape[0] < 4:
            padding = group[-1:].expand(4 - group.shape[0], -1, -1)
            group = torch.cat([group, padding], dim=0)
        averaged_c2w = torch.eye(4, dtype=c2ws_ref.dtype, device=c2ws_ref.device)
        averaged_rotation = R.from_matrix(group[:, :3, :3].detach().cpu().numpy()).mean().as_matrix()
        averaged_c2w[:3, :3] = torch.as_tensor(averaged_rotation, dtype=c2ws_ref.dtype, device=c2ws_ref.device)
        averaged_c2w[:3, 3] = group[:, :3, 3].mean(dim=0)
        averaged_c2ws_ref.append(averaged_c2w)
    averaged_c2ws_ref = torch.stack(averaged_c2ws_ref)
    averaged_target_intrinsics = _average_temporal_intrinsics(target_intrinsics, skip_vae_check)

    item["w2cs"] = invert_SE3(averaged_c2ws_ref)
    item["Ks"] = _intrinsics_to_K(averaged_target_intrinsics)

    neighbor_c2ws_world[:, :3, 3] = neighbor_c2ws_world[:, :3, 3] * scale
    neighbor_c2ws_ref = ref_w2c @ neighbor_c2ws_world
    item["neighbor_w2cs"] = invert_SE3(neighbor_c2ws_ref)
    item["neighbor_Ks"] = _intrinsics_to_K(neighbor_intrinsics)

    item["camera_rays"] = _camera_rays_for_intrinsics(
        averaged_target_intrinsics,
        averaged_c2ws_ref,
        image_height,
        image_width,
    )

    return item


def compute_camera_rays(
    transforms: dict,
    frame_indices: List[int],
    neighbor_indices: List[int],
    scale: float,
    image_shape: tuple[int, int],  # (H, W)
    skip_vae_check: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper that builds c2w tensors and per-frame intrinsics from transforms metadata."""
    H, W = image_shape
    frames = transforms["frames"]
    c2ws_world = torch.stack([torch.FloatTensor(frames[idx]["transform_matrix"]) for idx in frame_indices])
    neighbor_c2ws_world = torch.stack([torch.FloatTensor(frames[idx]["transform_matrix"]) for idx in neighbor_indices])
    target_intrinsics = torch.stack([_frame_intrinsics(transforms, frames[idx]) for idx in frame_indices])
    neighbor_intrinsics = torch.stack([_frame_intrinsics(transforms, frames[idx]) for idx in neighbor_indices])
    return compute_camera_conditioning(
        intrinsics=transforms,
        c2ws_world=c2ws_world,
        neighbor_c2ws_world=neighbor_c2ws_world,
        scale=scale,
        image_height=H,
        image_width=W,
        skip_vae_check=skip_vae_check,
        target_intrinsics=target_intrinsics,
        neighbor_intrinsics=neighbor_intrinsics,
    )


def visualize_inference_pairs(
    inference_items: List[Tuple[Any, InferencePair]],
    get_total_frames: callable,
    output_path: str | Path | None = None,
    title: str | None = None,
    selection_mode: NeighborSelectionMode | None = None,
) -> None:
    """
    Visualize train/test splits for inference pairs.

    Args:
        inference_items: List of (scene_key, InferencePair) tuples
        get_total_frames: Callable that takes scene_key and returns total frame count
        output_path: Path to save the figure (if None, displays interactively)
        title: Custom title for the plot
        selection_mode: Train selection mode for title
    """
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group inference items by scene
    scene_data = defaultdict(lambda: {"pairs": [], "total_frames": 0})
    for scene_key, pair in inference_items:
        # Get scene_id from pair if available, otherwise use scene_key
        scene_id = pair.scene_id if pair.scene_id else str(scene_key)
        scene_data[scene_id]["pairs"].append(pair)
        scene_data[scene_id]["total_frames"] = get_total_frames(scene_key)

    n_scenes = len(scene_data)
    if n_scenes == 0:
        print("No scenes to visualize")
        return

    fig, ax = plt.subplots(figsize=(14, max(4, 0.5 * n_scenes)))

    bar_height = 0.6
    train_color = "#2ECC40"  # Green
    blue_shades = ["#0074D9", "#7FDBFF", "#001f3f", "#39CCCC"]
    bg_color = "#CCCCCC"

    labels = []
    for y_pos, (scene_id, data) in enumerate(sorted(scene_data.items())):
        total_frames = data["total_frames"]
        pairs = data["pairs"]

        # Draw background (full frame range)
        ax.barh(y_pos, total_frames, height=bar_height, color=bg_color, left=0)

        # Draw test chunks with different blue shades
        for pair in pairs:
            test_indices = pair.test_indices if not pair.reversed else pair.test_indices[::-1]
            chunk_color = blue_shades[pair.chunk_idx % len(blue_shades)]
            sorted_test = sorted(test_indices)
            if sorted_test:
                seg_start = sorted_test[0]
                seg_end = sorted_test[0]
                for idx in sorted_test[1:]:
                    if idx == seg_end + 1:
                        seg_end = idx
                    else:
                        ax.barh(
                            y_pos,
                            seg_end - seg_start + 1,
                            height=bar_height,
                            color=chunk_color,
                            left=seg_start,
                            alpha=0.8,
                        )
                        seg_start = idx
                        seg_end = idx
                ax.barh(y_pos, seg_end - seg_start + 1, height=bar_height, color=chunk_color, left=seg_start, alpha=0.8)

        # Draw selected train indices on top (green)
        for pair in pairs:
            sorted_train = sorted(pair.neighbor_indices)
            if sorted_train:
                seg_start = sorted_train[0]
                seg_end = sorted_train[0]
                for idx in sorted_train[1:]:
                    if idx == seg_end + 1:
                        seg_end = idx
                    else:
                        ax.barh(y_pos, seg_end - seg_start + 1, height=bar_height, color=train_color, left=seg_start)
                        seg_start = idx
                        seg_end = idx
                ax.barh(y_pos, seg_end - seg_start + 1, height=bar_height, color=train_color, left=seg_start)

        labels.append(f"{scene_id[:12]} ({len(pairs)}p)")

    ax.set_yticks(range(n_scenes))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Frame Index")
    ax.invert_yaxis()

    if title is None:
        mode_str = selection_mode.value.upper() if selection_mode else "UNKNOWN"
        title = f"Train (green) / Test chunks (blue) - {mode_str}"
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    plt.close()
