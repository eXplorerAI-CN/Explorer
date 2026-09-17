# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from data_processing.scene_utils import load_scene_transforms_from_dir, load_scene_transforms_from_zip


def so3_relative_angle(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Compute the angle between 2 rotation matrices.
    Args:
        R1: the 1st rotation matrix with shape (B, 3, 3)
        R2: the 2nd rotation matrix with shape (B, 3, 3)
    Returns:
        The rotation matrix angle in radians with shape (B,)
    """
    # (Trace(R1.T @ R2) - 1)/2
    prod = np.matmul(np.swapaxes(R1, -2, -1), R2)
    cos = (np.trace(prod, axis1=-2, axis2=-1) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    diff = np.abs(np.arccos(cos))
    return diff


def find_farthest_camera_pair(extrinsics_c2w, lambda_t: float = 1.0, normalize: bool = True):
    """Find the farthest camera pose pair in all poses.
    Returns:
        farthest_pair: the tuple of the farthest camera pair indices
        max_dist: the distance between the farthest camera pair
    """
    rs = extrinsics_c2w[:, :3, :3]
    ts = extrinsics_c2w[:, :3, 3]
    if normalize:
        avg_scale = np.mean(np.linalg.norm(ts, axis=1))
        ts = ts / avg_scale

    t_dists = np.linalg.norm(ts[:, None, :] - ts[None, :, :], axis=-1)
    r_dists = so3_relative_angle(rs[:, None, ...], rs[None, ...])
    r_dists = np.rad2deg(r_dists) / 180.0
    dists = r_dists + lambda_t * t_dists

    max_dist = dists.max()
    overall_max_index = np.argmax(dists)
    farthest_pair = np.unravel_index(overall_max_index, dists.shape)

    return farthest_pair, max_dist


def _pose_distance(ref_idx: int, extrinsics_c2w: np.ndarray, lambda_t: float = 1.0, normalize: bool = True):
    """
    Calculate the distance between the reference camera and all cameras.
    The distance function is defined as: dist = r_dist + lambda_t * t_dist
    Args:
        ref_idx (int): The index of the reference camera.
        extrinsics_c2w (np.ndarray): (N, 4, 4) Camera extrinsics.
        lambda_t (float): The coefficient of the translation distance.
        normalize (bool): To normalize the camera translation or not.
    Return:
        dists: A list of distances between the reference camear and all cameras.
    """
    pos = extrinsics_c2w[:, :3, 3]
    if normalize:
        avg_scale = np.mean(np.linalg.norm(pos, axis=1))
        pos = pos / avg_scale

    t_dists = np.linalg.norm(pos - pos[ref_idx], axis=1)
    r_dists = so3_relative_angle(extrinsics_c2w[:, :3, :3], extrinsics_c2w[ref_idx, None, :3, :3])
    r_dists = np.rad2deg(r_dists) / 180.0
    dists = r_dists + lambda_t * t_dists
    return dists


def rank_views_by_pose_distance(
    ref_idx: int, extrinsics_c2w: np.ndarray, lambda_t: float = 1.0, normalize: bool = True
):
    """
    Rank views by pose distance.
    Args:
        ref_idx (int): The index of the reference camera.
        extrinsics_c2w (np.ndarray): (N, 4, 4) Camera extrinsics.
        lambda_t (float): The coefficient of the translation distance.
        normalize (bool): To normalize the camera translation or not.
    Return:
        sorted_idxs: A list of sorted camera indices from near to far.
    """
    dists = _pose_distance(ref_idx, extrinsics_c2w, lambda_t, normalize)
    sorted_idxs = np.argsort(dists)
    return sorted_idxs


def order_indices_by_farthest_sampling(init_idx, extrinsics_c2w, valid_mask):
    """
    Performs Farthest Point Sampling on camera poses. For a given initial camera index and set of valid indices marked by valid_mask,
    return an ordered list of camera indices such that the camera poses will be as far as possible from each other.
    Args:
        init_idx (int): The index of the starting camera pose to do farthest sampling.
        extrinsics_c2w (np.ndarray): (N, 4, 4) Camera extrinsics.
        valid_mask (np.array): (N, ) boolean array to filter out unused camera poses.
    Returns:
        sampled_indices (np.array): (N_, ) The indices of selected camera poses
    """
    num_poses = extrinsics_c2w.shape[0]
    num_samples = valid_mask.sum()

    # Initialize sampled_indices with the first point (can be random)
    sampled_indices = np.zeros(num_samples, dtype=int)
    sampled_indices[0] = init_idx

    # Initialize min_distances to minus infinity for all points
    min_distances = np.full(num_poses, -np.inf)

    # Calculate initial distances from the first sampled point
    dists = _pose_distance(ref_idx=sampled_indices[0], extrinsics_c2w=extrinsics_c2w)
    min_distances = dists
    min_distances[~valid_mask] = -np.inf

    for i in range(1, num_samples):
        # Find the point with the maximum minimum distance
        farthest_idx = np.argmax(min_distances)
        sampled_indices[i] = farthest_idx

        # Update min_distances based on the newly sampled point
        curr_dists = _pose_distance(ref_idx=sampled_indices[i], extrinsics_c2w=extrinsics_c2w)
        curr_dists[~valid_mask] = -np.inf
        min_distances = np.minimum(min_distances, curr_dists)

    return sampled_indices


def load_scene_transforms(dataset_path: Path) -> dict:
    if dataset_path.suffix == ".zip":
        with zipfile.ZipFile(dataset_path, "r") as zf:
            data, _ = load_scene_transforms_from_zip(zf, dataset_path.stem)
    else:
        data = load_scene_transforms_from_dir(dataset_path)
    return data


def write_half_covisibility_sampled_indices(dataset_path: Path, output_path: Path) -> None:
    data = load_scene_transforms(dataset_path)
    extrinsics_c2w = np.array([np.array(data["frames"][i]["transform_matrix"]) for i in range(len(data["frames"]))])

    farthest_pair, _ = find_farthest_camera_pair(extrinsics_c2w=extrinsics_c2w)

    rank_views_first_half = rank_views_by_pose_distance(ref_idx=farthest_pair[0], extrinsics_c2w=extrinsics_c2w)
    valid_mask_first_half = np.full(len(data["frames"]), False)
    valid_mask_second_half = np.full(len(data["frames"]), False)
    valid_mask_first_half[rank_views_first_half[: len(data["frames"]) // 2]] = True
    valid_mask_second_half[rank_views_first_half[len(data["frames"]) // 2 :]] = True

    output_path.mkdir(parents=True, exist_ok=True)

    init_idx = farthest_pair[0]
    sampled_indices = order_indices_by_farthest_sampling(
        init_idx=init_idx, extrinsics_c2w=extrinsics_c2w, valid_mask=valid_mask_first_half
    )
    save_path = output_path / "half_covisibility_sampled_indices_0.json"
    with open(save_path, "w") as json_file:
        json.dump(sampled_indices.tolist(), json_file, indent=4)
    print(f"Save the first half to {save_path}")

    init_idx = farthest_pair[1]
    sampled_indices = order_indices_by_farthest_sampling(
        init_idx=init_idx, extrinsics_c2w=extrinsics_c2w, valid_mask=valid_mask_second_half
    )
    save_path = output_path / "half_covisibility_sampled_indices_1.json"
    with open(save_path, "w") as json_file:
        json.dump(sampled_indices.tolist(), json_file, indent=4)
    print(f"Save the second half to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A simple script to divide cameras into two groups and sort them according to the distance."
    )
    parser.add_argument("--dataset_path", type=Path, required=True, help="The path of the scene.")
    parser.add_argument("--output_path", type=Path, required=True, help="The path to save the sampled indices.")
    args = parser.parse_args()
    write_half_covisibility_sampled_indices(args.dataset_path, args.output_path)


if __name__ == "__main__":
    main()
