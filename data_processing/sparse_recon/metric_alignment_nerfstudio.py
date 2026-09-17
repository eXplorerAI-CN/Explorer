# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Metric alignment for nerfstudio datasets without images.bin.

Uses transforms.json for poses and projects all 3D points to images
instead of relying on 2D-3D correspondences from images.bin.

Usage:
    python \
        metric_alignment_nerfstudio.py \
        --scene_dir /path/to/scene/nerfstudio \
        --output_dir ./output
"""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from data_processing.sparse_recon.metric_alignment import (
    MoGeModelWrapper,
    create_error_heatmap,
    plot_depth_correlation,
    read_colmap_points3D_binary,
    sanity_checks,
    solve_scale_factor,
    visualize_depth_comparison,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def read_nerfstudio_transforms(transforms_path):
    """Read camera intrinsics, poses, and applied_transform from nerfstudio transforms.json."""
    with open(transforms_path) as f:
        data = json.load(f)

    # Extract camera intrinsics
    intrinsics = {
        "w": data["w"],
        "h": data["h"],
        "fl_x": data["fl_x"],
        "fl_y": data["fl_y"],
        "cx": data["cx"],
        "cy": data["cy"],
    }

    # Extract applied_transform if present (used to transform COLMAP points to match poses)
    applied_transform = None
    if "applied_transform" in data:
        applied_transform = np.array(data["applied_transform"])
        # Ensure it's 4x4
        if applied_transform.shape == (3, 4):
            full_transform = np.eye(4)
            full_transform[:3, :] = applied_transform
            applied_transform = full_transform

    # Extract frames with poses
    frames = []
    for frame in data["frames"]:
        c2w = np.array(frame["transform_matrix"])

        # Compute W2C and apply transforms (following 3dgrut convention)
        w2c = np.zeros((4, 4), dtype=np.float32)
        w2c[:3, :3] = c2w[:3, :3].T
        w2c[:3, 3] = -c2w[:3, :3].T @ c2w[:3, 3]
        w2c[3, 3] = 1.0

        # Apply the inverse applied_transform to "undo" nerfstudio's transform
        if applied_transform is not None:
            w2c = w2c @ applied_transform

        # Convert from OpenGL (Y-up, -Z forward) to OpenCV (Y-down, +Z forward)
        w2c[1:3, :] *= -1

        frames.append(
            {
                "file_path": frame["file_path"],
                "c2w": np.linalg.inv(w2c),
                "w2c": w2c,
                "R": w2c[:3, :3],
                "t": w2c[:3, 3],
            }
        )

    # Return None for applied_transform since it's already applied to poses
    return intrinsics, frames, None


def project_points_to_frame(points3D, frame, intrinsics):
    """Project 3D points to image using frame pose and intrinsics."""
    R = frame["R"]
    t = frame["t"].reshape(3, 1)

    # Transform to camera coordinates
    points_cam = R @ points3D.T + t  # 3 x N

    # Filter points behind camera
    valid_depth = points_cam[2] > 0.1  # minimum depth threshold

    if valid_depth.sum() == 0:
        return np.array([]), np.array([]), np.array([])

    points_cam_valid = points_cam[:, valid_depth]
    depths = points_cam_valid[2]

    # Project to image
    fx, fy = intrinsics["fl_x"], intrinsics["fl_y"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    x = points_cam_valid[0] / points_cam_valid[2] * fx + cx
    y = points_cam_valid[1] / points_cam_valid[2] * fy + cy

    # Filter points outside image bounds
    w, h = intrinsics["w"], intrinsics["h"]
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    return x[valid], y[valid], depths[valid]


def extract_projected_depths(frames, intrinsics, points3D, image_dir, downsample_factor=1.0):
    """Extract COLMAP depths by projecting all 3D points to each frame."""
    correspondences = []

    for frame in frames:
        # Get image path
        file_path = frame["file_path"]
        if file_path.startswith("images/"):
            image_name = file_path.split("/")[-1]
        else:
            image_name = file_path

        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue

        # Project all 3D points to this frame
        x, y, depths_colmap = project_points_to_frame(points3D, frame, intrinsics)

        if len(depths_colmap) == 0:
            continue

        xys = np.stack([x, y], axis=1)

        correspondences.append(
            {
                "image_name": image_name,
                "xys": xys,
                "depths_colmap": depths_colmap,
                "image_path": image_path,
                "camera": {
                    "width": intrinsics["w"],
                    "height": intrinsics["h"],
                    "params": np.array([intrinsics["fl_x"], intrinsics["fl_y"], intrinsics["cx"], intrinsics["cy"]]),
                },
            }
        )

    return correspondences


def extract_monodepth_depths(correspondences, monodepth_model, downsample_factor=1.0):
    """Extract monodepth predictions at projected point locations."""
    for corr in correspondences:
        image_path = corr["image_path"]

        input_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        input_image = torch.tensor(input_image / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)

        pred_depth, mask, _ = monodepth_model(input_image)
        pred_depth = pred_depth.cpu().numpy()
        mask = mask.cpu().numpy()

        if mask.ndim > 2:
            mask = mask.squeeze()

        # Sample depths at point locations
        # (xys are already in downsampled image coordinates since intrinsics were scaled)
        xys = corr["xys"]
        depth_h, depth_w = pred_depth.shape
        x_idx = np.clip(xys[:, 0].astype(int), 0, depth_w - 1)
        y_idx = np.clip(xys[:, 1].astype(int), 0, depth_h - 1)

        depths_metric = pred_depth[y_idx, x_idx]
        mask_sampled = mask[y_idx, x_idx]

        corr["depths_metric"] = depths_metric
        corr["mask"] = mask_sampled

    return correspondences


def align_nerfstudio_to_metric_scale(
    scene_dir,
    output_dir=None,
    num_images=None,
    debug=True,
    downsample_factor=1.0,
):
    """
    Align nerfstudio scene to metric scale.

    Args:
        scene_dir: Path to nerfstudio scene directory (contains transforms.json)
        output_dir: Output directory for results
        num_images: Limit number of images to process
        debug: Enable debug visualizations
        downsample_factor: Image downsample factor
    """
    scene_dir = Path(scene_dir)

    # Find transforms.json
    transforms_path = scene_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found in {scene_dir}")

    # Find colmap directory for points3D.bin
    colmap_dir = scene_dir / "colmap" / "sparse" / "0"
    if not colmap_dir.exists():
        colmap_dir = scene_dir / "colmap"

    points3D_path = colmap_dir / "points3D.bin"
    if not points3D_path.exists():
        raise FileNotFoundError(f"points3D.bin not found in {colmap_dir}")

    # Determine image directory
    if downsample_factor == 1.0:
        image_dir = scene_dir / "images"
    else:
        image_dir = scene_dir / f"images_{int(downsample_factor)}"

    if not image_dir.exists():
        # Try parent directory
        image_dir = (
            scene_dir.parent / "images"
            if downsample_factor == 1.0
            else scene_dir.parent / f"images_{int(downsample_factor)}"
        )

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Setup debug directory
    if debug and output_dir:
        debug_dir = Path(output_dir) / "debug_visualizations"
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug visualizations will be saved to: {debug_dir}")

    # Load data
    logger.info("Loading nerfstudio transforms...")
    intrinsics, frames, applied_transform = read_nerfstudio_transforms(transforms_path)
    logger.info(f"Loaded {len(frames)} frames")
    logger.info(f"Original intrinsics: {intrinsics['w']}x{intrinsics['h']}, fx={intrinsics['fl_x']:.1f}")

    # Scale intrinsics for downsampled images
    if downsample_factor != 1.0:
        intrinsics["w"] = int(intrinsics["w"] / downsample_factor)
        intrinsics["h"] = int(intrinsics["h"] / downsample_factor)
        intrinsics["fl_x"] = intrinsics["fl_x"] / downsample_factor
        intrinsics["fl_y"] = intrinsics["fl_y"] / downsample_factor
        intrinsics["cx"] = intrinsics["cx"] / downsample_factor
        intrinsics["cy"] = intrinsics["cy"] / downsample_factor
        logger.info(f"Scaled intrinsics: {intrinsics['w']}x{intrinsics['h']}, fx={intrinsics['fl_x']:.1f}")

    logger.info("Loading COLMAP points3D...")
    point_ids, points_xyz, points_rgb, points_error = read_colmap_points3D_binary(str(points3D_path))
    logger.info(f"Loaded {len(points_xyz)} 3D points")

    if num_images is not None:
        frames = frames[:num_images]

    # Extract projected depths
    logger.info(f"Projecting 3D points to {len(frames)} frames...")
    correspondences = extract_projected_depths(frames, intrinsics, points_xyz, str(image_dir), downsample_factor)
    logger.info(f"Found {len(correspondences)} frames with valid projections")

    # Run monodepth
    logger.info("\nRunning monodepth...")
    monodepth_model = MoGeModelWrapper()
    correspondences = extract_monodepth_depths(correspondences, monodepth_model, downsample_factor)

    # Filter out inf/nan values from metric depths
    for corr in correspondences:
        if corr.get("depths_metric") is not None:
            valid = np.isfinite(corr["depths_metric"])
            corr["depths_metric"] = corr["depths_metric"][valid]
            corr["depths_colmap"] = corr["depths_colmap"][valid]
            corr["xys"] = corr["xys"][valid]
            if corr.get("mask") is not None:
                corr["mask"] = corr["mask"][valid]

    # Debug visualizations
    if debug and output_dir:
        logger.info("\nCreating depth comparison visualizations...")
        num_viz = min(10, len([c for c in correspondences if c.get("depths_metric") is not None]))
        # Pass downsample_factor=1.0 since xys are already in downsampled coordinates
        # (we project using scaled intrinsics, unlike metric_alignment.py which uses COLMAP's full-res xys)
        visualize_depth_comparison(correspondences, debug_dir, num_samples=num_viz, downsample_factor=1.0)

    # Solve for scale
    logger.info("\nSolving for scale factor...")
    scale, stats = solve_scale_factor(correspondences)

    logger.info("\nResults:")
    logger.info(f"  Scale factor: {scale:.6f}")
    logger.info(f"  RMSE: {stats['rmse']:.4f} (weighted: {stats['weighted_rmse']:.4f})")
    logger.info(f"  Mean error: {stats['mean_error']:.4f} (weighted: {stats['weighted_mean_error']:.4f})")
    logger.info(f"  Median error: {stats['median_error']:.4f}")
    logger.info(f"  Correspondences: {stats['num_correspondences']}")

    # More debug visualizations
    if debug and output_dir:
        logger.info("\nCreating correlation plots...")
        plot_depth_correlation(correspondences, scale, debug_dir / "depth_correlation.png")
        create_error_heatmap(correspondences, scale, debug_dir)
        sanity_checks(correspondences, scale, stats)

    # Save scale info
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "scale_info.txt", "w") as f:
            f.write(f"Scale factor: {scale}\n")
            f.write(f"RMSE: {stats['rmse']:.6f}\n")
            f.write(f"Median error: {stats['median_error']:.6f}\n")
            f.write(f"Num correspondences: {stats['num_correspondences']}\n")

    return scale, stats, correspondences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metric alignment for nerfstudio datasets (without images.bin)")
    parser.add_argument(
        "--scene_dir", required=True, help="Path to nerfstudio scene directory (contains transforms.json)"
    )
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    parser.add_argument("--num_images", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--no_debug", action="store_true", help="Disable debug visualizations")
    parser.add_argument("--downsample_factor", type=float, default=1.0, help="Image downsample factor")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join("./debug_nerfstudio", Path(args.scene_dir).name)

    scale, stats, correspondences = align_nerfstudio_to_metric_scale(
        args.scene_dir,
        args.output_dir,
        args.num_images,
        debug=not args.no_debug,
        downsample_factor=args.downsample_factor,
    )
