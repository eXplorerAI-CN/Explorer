# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from threedgrut.datasets.utils import (
    Image,
    qvec_to_so3,
    read_colmap_extrinsics_binary,
    read_colmap_intrinsics_binary,
    read_next_bytes,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class MoGeModelWrapper:
    def __init__(self, model_variant="Ruicheng/moge-2-vitl-normal", device="cuda"):
        if "MOGE_MODEL_PATH" in os.environ and Path(os.environ["MOGE_MODEL_PATH"]).exists():
            local_model_path = os.environ["MOGE_MODEL_PATH"]
            logger.info(f"Loading MoGe from local path: {local_model_path}")
            model = MoGeModel.from_pretrained(local_model_path).to(device)
        else:
            logger.info(f"Loading MoGe from HuggingFace: {model_variant}")
            model = MoGeModel.from_pretrained(model_variant).to(device)
        self.model = model

    @torch.no_grad()
    def __call__(self, rgb_origin):
        output = self.model.infer(rgb_origin)
        sky_mask = output["depth"] == torch.inf
        return output["depth"], output["mask"] & ~sky_mask, output["intrinsics"]


def read_colmap_points3D_binary(path_to_model_file):
    point_ids, xyzs, rgbs, errors = [], [], [], []
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point_ids.append(int(point_properties[0]))
            xyzs.append(point_properties[1:4])
            rgbs.append(point_properties[4:7])
            errors.append(point_properties[7])
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(8 * track_length, 1)

    return (
        np.array(point_ids, dtype=np.int64),
        np.array(xyzs, dtype=np.float64),
        np.array(rgbs, dtype=np.int32),
        np.array(errors, dtype=np.float64).reshape(-1, 1),
    )


def project_points_to_image(points3D, qvec, tvec, camera_params, width, height):
    """Project 3D points to image and return depths and pixel coordinates."""
    R = qvec_to_so3(qvec)
    t = tvec.reshape(3, 1)

    # Transform to camera coordinates
    points_cam = R @ points3D.T + t

    # Filter points behind camera
    valid_depth = points_cam[2] > 0
    points_cam = points_cam[:, valid_depth]

    if points_cam.shape[1] == 0:
        return np.array([]), np.array([]), np.array([])

    depths = points_cam[2]

    # Project to image (assuming PINHOLE model)
    fx, fy, cx, cy = camera_params[:4]
    x = points_cam[0] / points_cam[2] * fx + cx
    y = points_cam[1] / points_cam[2] * fy + cy

    # Filter points outside image bounds
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    return x[valid], y[valid], depths[valid]


def extract_colmap_depths(images, cameras, point3D_ids, points3D_xyz, image_dir):
    """Extract COLMAP depths for visible 3D points in each image."""
    # Create mapping from point3D_id to array index
    id_to_idx = {pid: idx for idx, pid in enumerate(point3D_ids)}

    correspondences = []

    for img in images:
        # Get valid 3D point observations
        valid_mask = img.point3D_ids != -1
        if valid_mask.sum() == 0:
            continue

        img_point3D_ids = img.point3D_ids[valid_mask]
        xys = img.xys[valid_mask]

        # Get 3D points using ID mapping
        valid_points = []
        valid_xys = []
        for i, pid in enumerate(img_point3D_ids):
            if pid in id_to_idx:
                valid_points.append(points3D_xyz[id_to_idx[pid]])
                valid_xys.append(xys[i])

        if len(valid_points) == 0:
            continue

        points = np.array(valid_points)
        xys = np.array(valid_xys)

        # Compute depths
        R = qvec_to_so3(img.qvec)
        t = img.tvec.reshape(3, 1)
        points_cam = R @ points.T + t
        depths = points_cam[2]

        # Store correspondences
        camera = cameras[img.camera_id]
        correspondences.append(
            {
                "image_name": img.name,
                "xys": xys,
                "depths_colmap": depths,
                "camera": camera,
                "image_path": os.path.join(image_dir, img.name),
            }
        )

    return correspondences


def extract_monodepth_depths(correspondences, monodepth_model, downsample_factor=1.0):
    """Extract Monodepth depth predictions at COLMAP point locations."""
    for corr in correspondences:
        image_path = corr["image_path"]
        if not os.path.exists(image_path):
            corr["depths_metric"] = None
            corr["mask"] = None
            continue

        input_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        input_image = torch.tensor(input_image / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)

        pred_depth, mask, _ = monodepth_model(input_image)
        pred_depth = pred_depth.cpu().numpy()
        mask = mask.cpu().numpy()

        # Squeeze mask if it has extra dimensions
        if mask.ndim > 2:
            mask = mask.squeeze()

        # Sample depths and mask at point locations (scale coordinates to match downsampled image)
        xys = corr["xys"] / downsample_factor
        depth_h, depth_w = pred_depth.shape
        x_idx = np.clip(xys[:, 0].astype(int), 0, depth_w - 1)
        y_idx = np.clip(xys[:, 1].astype(int), 0, depth_h - 1)

        depths_metric = pred_depth[y_idx, x_idx]
        mask_sampled = mask[y_idx, x_idx]

        corr["depths_metric"] = depths_metric
        corr["mask"] = mask_sampled

    return correspondences


def solve_scale_factor(correspondences):
    """Solve for scale factor: metric_depth = scale * colmap_depth using mask-weighted L1 loss.

    Args:
        correspondences: List of correspondence dictionaries
    """
    colmap_depths = []
    metric_depths = []
    masks = []

    for corr in correspondences:
        if corr["depths_metric"] is None:
            continue
        colmap_depths.append(corr["depths_colmap"])
        metric_depths.append(corr["depths_metric"])
        if corr.get("mask") is not None:
            masks.append(corr["mask"])
        else:
            masks.append(np.ones_like(corr["depths_colmap"]))

    colmap_depths = np.concatenate(colmap_depths)
    metric_depths = np.concatenate(metric_depths)
    masks = np.concatenate(masks)

    # Filter outliers
    valid = (metric_depths > 0) & (colmap_depths > 0)
    colmap_depths = colmap_depths[valid]
    metric_depths = metric_depths[valid]
    masks = masks[valid]

    # Normalize masks to have mean 1 (so they act as weights)
    if masks.sum() > 0:
        masks = masks / masks.mean()
    else:
        masks = np.ones_like(colmap_depths)

    # Mask-weighted L1 loss optimization
    # For L1 loss, the optimal scale is the weighted median of depth ratios
    ratios = metric_depths / colmap_depths

    # Compute weighted median
    sorted_idx = np.argsort(ratios)
    sorted_ratios = ratios[sorted_idx]
    sorted_weights = masks[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2.0)
    scale = sorted_ratios[median_idx]

    # Compute statistics
    scaled_colmap = scale * colmap_depths
    errors = np.abs(scaled_colmap - metric_depths)

    # Compute both weighted and unweighted statistics
    weighted_mean_error = (masks * errors).sum() / masks.sum()
    weighted_rmse = np.sqrt((masks * errors**2).sum() / masks.sum())

    stats = {
        "scale": scale,
        "mean_error": errors.mean(),
        "weighted_mean_error": weighted_mean_error,
        "median_error": np.median(errors),
        "rmse": np.sqrt((errors**2).mean()),
        "weighted_rmse": weighted_rmse,
        "num_correspondences": len(errors),
        "mean_mask": masks.mean(),
        "median_mask": np.median(masks),
    }

    return scale, stats


def visualize_depth_comparison(correspondences, output_dir, num_samples=5, downsample_factor=1.0):
    """Visualize COLMAP vs Monodepth depth maps side by side."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, corr in enumerate(correspondences[:num_samples]):
        if corr["depths_metric"] is None:
            continue

        # Load image
        rgb = cv2.imread(corr["image_path"])[:, :, ::-1]

        # Scale xy coordinates to match downsampled image
        xys_scaled = corr["xys"] / downsample_factor

        # Create depth maps with points marked (3x2 grid if mask available)
        has_mask = corr.get("mask") is not None
        if has_mask:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Original image
        ax_img = axes[0, 0]
        ax_img.imshow(rgb)
        ax_img.scatter(xys_scaled[:, 0], xys_scaled[:, 1], c="red", s=1, alpha=0.5)
        ax_img.set_title(f"Image: {corr['image_name']} (downsampled {downsample_factor}x)")
        ax_img.axis("off")

        # COLMAP depths (scattered points)
        ax_colmap = axes[0, 1]
        scatter1 = ax_colmap.scatter(
            xys_scaled[:, 0],
            xys_scaled[:, 1],
            c=corr["depths_colmap"],
            s=10,
            cmap="plasma",
        )
        ax_colmap.set_title("COLMAP Depths (sparse)")
        ax_colmap.set_xlim(0, rgb.shape[1])
        ax_colmap.set_ylim(rgb.shape[0], 0)
        plt.colorbar(scatter1, ax=ax_colmap)

        # Metric depths (scattered points)
        ax_metric = axes[1, 0]
        scatter2 = ax_metric.scatter(
            xys_scaled[:, 0],
            xys_scaled[:, 1],
            c=corr["depths_metric"],
            s=10,
            cmap="plasma",
        )
        ax_metric.set_title("Metric Depths (sampled)")
        ax_metric.set_xlim(0, rgb.shape[1])
        ax_metric.set_ylim(rgb.shape[0], 0)
        plt.colorbar(scatter2, ax=ax_metric)

        # Depth difference
        ax_diff = axes[1, 1]
        diff = corr["depths_metric"] - corr["depths_colmap"]
        scatter3 = ax_diff.scatter(
            xys_scaled[:, 0],
            xys_scaled[:, 1],
            c=diff,
            s=10,
            cmap="RdBu",
            vmin=-5,
            vmax=5,
        )
        ax_diff.set_title("Difference (Metric - COLMAP)")
        ax_diff.set_xlim(0, rgb.shape[1])
        ax_diff.set_ylim(rgb.shape[0], 0)
        plt.colorbar(scatter3, ax=ax_diff)

        # Mask map (if available)
        if has_mask:
            ax_conf = axes[0, 2]
            scatter4 = ax_conf.scatter(
                xys_scaled[:, 0],
                xys_scaled[:, 1],
                c=corr["mask"],
                s=10,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            ax_conf.set_title(f"Metric Mask (mean: {corr['mask'].mean():.3f})")
            ax_conf.set_xlim(0, rgb.shape[1])
            ax_conf.set_ylim(rgb.shape[0], 0)
            plt.colorbar(scatter4, ax=ax_conf)

        plt.tight_layout()
        plt.savefig(output_dir / f"depth_comparison_{idx:03d}.png", dpi=150)
        plt.close()
        logger.info(f"  Saved depth comparison {idx + 1}/{num_samples}")


def plot_depth_correlation(correspondences, scale, output_path):
    """Create scatter plot of COLMAP vs Metric depths with fitted scale."""
    colmap_depths = []
    metric_depths = []
    masks = []

    for corr in correspondences:
        if corr["depths_metric"] is not None:
            colmap_depths.append(corr["depths_colmap"])
            metric_depths.append(corr["depths_metric"])
            if corr.get("mask") is not None:
                masks.append(corr["mask"])
            else:
                masks.append(np.ones_like(corr["depths_colmap"]))

    colmap_depths = np.concatenate(colmap_depths)
    metric_depths = np.concatenate(metric_depths)
    masks = np.concatenate(masks)

    fig, axes = plt.subplots(1, 2, figsize=(14, 12))

    # Scatter plot with fit line (colored by mask)
    scatter = axes[0].scatter(
        colmap_depths,
        metric_depths,
        c=masks,
        cmap="viridis",
        alpha=0.3,
        s=1,
        vmin=0,
        vmax=1,
    )
    axes[0].plot(
        [0, colmap_depths.max()],
        [0, colmap_depths.max() * scale],
        "r-",
        linewidth=2,
        label=f"Fitted scale={scale:.3f}",
    )
    axes[0].plot(
        [0, colmap_depths.max()],
        [0, colmap_depths.max()],
        "k--",
        linewidth=1,
        alpha=0.5,
        label="Scale=1.0",
    )
    axes[0].set_xlabel("COLMAP Depth")
    axes[0].set_ylabel("Metric Depth")
    axes[0].set_title("Depth Correlation (colored by mask)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label="Mask")

    # Residuals plot (colored by mask)
    residuals = metric_depths - scale * colmap_depths
    scatter2 = axes[1].scatter(
        colmap_depths,
        residuals,
        c=masks,
        cmap="viridis",
        alpha=0.3,
        s=1,
        vmin=0,
        vmax=1,
    )
    axes[1].axhline(y=0, color="r", linestyle="--")
    axes[1].set_xlabel("COLMAP Depth")
    axes[1].set_ylabel("Residual (Metric - Scaled COLMAP)")
    axes[1].set_title("Residuals vs Depth (colored by mask)")
    axes[1].grid(alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label="Mask")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("  Saved depth correlation plot")


def visualize_point_cloud_comparison(points_before, points_after, output_path, max_points=10000):
    """Create side-by-side visualization of point clouds."""
    # Subsample for visualization
    if len(points_before) > max_points:
        idx = np.random.choice(len(points_before), max_points, replace=False)
        points_before_sub = points_before[idx]
        points_after_sub = points_after[idx]
    else:
        points_before_sub = points_before
        points_after_sub = points_after

    fig = plt.figure(figsize=(16, 7))

    # Before scaling
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        points_before_sub[:, 0],
        points_before_sub[:, 1],
        points_before_sub[:, 2],
        s=0.1,
        alpha=0.5,
    )
    ax1.set_title("Before Scaling")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # After scaling
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        points_after_sub[:, 0],
        points_after_sub[:, 1],
        points_after_sub[:, 2],
        s=0.1,
        alpha=0.5,
    )
    ax2.set_title("After Metric Scaling")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    # Print statistics
    logger.info("\n  Point cloud statistics:")
    logger.info(
        f"    Before - Range: X=[{points_before[:, 0].min():.2f}, {points_before[:, 0].max():.2f}], "
        f"Y=[{points_before[:, 1].min():.2f}, {points_before[:, 1].max():.2f}], "
        f"Z=[{points_before[:, 2].min():.2f}, {points_before[:, 2].max():.2f}]"
    )
    logger.info(
        f"    After  - Range: X=[{points_after[:, 0].min():.2f}, {points_after[:, 0].max():.2f}], "
        f"Y=[{points_after[:, 1].min():.2f}, {points_after[:, 1].max():.2f}], "
        f"Z=[{points_after[:, 2].min():.2f}, {points_after[:, 2].max():.2f}]"
    )


def create_error_heatmap(correspondences, scale, output_dir):
    """Create heatmap showing errors per image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_errors = []

    for corr in correspondences:
        if corr["depths_metric"] is None:
            continue

        scaled_colmap = scale * corr["depths_colmap"]
        errors = np.abs(scaled_colmap - corr["depths_metric"])

        image_errors.append(
            {
                "name": corr["image_name"],
                "mean_error": errors.mean(),
                "median_error": np.median(errors),
                "max_error": errors.max(),
                "num_points": len(errors),
            }
        )

    if len(image_errors) == 0:
        logger.info("  No valid image errors to plot")
        return

    # Plot per-image error metrics
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    names = [e["name"] for e in image_errors]
    mean_errors = [e["mean_error"] for e in image_errors]

    ax.bar(range(len(names)), mean_errors)
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Per-Image Mean Absolute Error")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "per_image_errors.png", dpi=150)
    plt.close()
    logger.info("  Saved per-image error plots")


def sanity_checks(correspondences, scale, stats):
    """Run sanity checks on the alignment results."""
    logger.info("\n" + "=" * 50)
    logger.info("SANITY CHECKS")
    logger.info("=" * 50)

    # Check 1: Scale factor reasonableness
    if scale < 0.1 or scale > 100:
        logger.info(f"⚠️  WARNING: Scale factor seems unreasonable: {scale}")
    else:
        logger.info(f"✓ Scale factor is reasonable: {scale:.4f}")

    # Check 2: Correspondence coverage
    total_points = sum(len(c["depths_colmap"]) for c in correspondences if c["depths_metric"] is not None)
    if total_points < 1000:
        logger.info(f"⚠️  WARNING: Low number of correspondences: {total_points}")
    else:
        logger.info(f"✓ Good correspondence count: {total_points}")

    # Check 3: Error magnitude
    if stats["median_error"] > 1.0:  # More than 1 meter
        logger.info(f"⚠️  WARNING: High median error: {stats['median_error']:.3f}m")
    else:
        logger.info(f"✓ Median error acceptable: {stats['median_error']:.3f}m")

    # Check 4: Depth range consistency
    all_colmap = np.concatenate([c["depths_colmap"] for c in correspondences if c["depths_metric"] is not None])
    all_metric = np.concatenate([c["depths_metric"] for c in correspondences if c["depths_metric"] is not None])

    logger.info("\nDepth ranges:")
    logger.info(f"  COLMAP:        [{all_colmap.min():.2f}, {all_colmap.max():.2f}]")
    logger.info(f"  Metric:      [{all_metric.min():.2f}, {all_metric.max():.2f}]")
    logger.info(f"  Scaled COLMAP: [{(all_colmap * scale).min():.2f}, {(all_colmap * scale).max():.2f}]")

    # Check 5: Per-image coverage
    images_with_data = sum(1 for c in correspondences if c["depths_metric"] is not None)
    total_images = len(correspondences)
    coverage = images_with_data / total_images * 100 if total_images > 0 else 0

    if coverage < 50:
        logger.info(f"\n⚠️  WARNING: Low image coverage: {images_with_data}/{total_images} ({coverage:.1f}%)")
    else:
        logger.info(f"\n✓ Good image coverage: {images_with_data}/{total_images} ({coverage:.1f}%)")

    logger.info("=" * 50 + "\n")


def scale_colmap_reconstruction(scale, colmap_dir, output_dir):
    """Apply scale to COLMAP reconstruction and save."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    points3D_path = os.path.join(colmap_dir, "points3D.bin")
    images_path = os.path.join(colmap_dir, "images.bin")

    _, points_xyz, points_rgb, points_error = read_colmap_points3D_binary(points3D_path)

    images = read_colmap_extrinsics_binary(images_path)

    # Scale points and translations
    points_xyz_scaled = points_xyz * scale

    scaled_images = []
    for img in images:
        scaled_img = Image(
            id=img.id,
            qvec=img.qvec,
            tvec=img.tvec * scale,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,
        )
        scaled_images.append(scaled_img)

    # Save scaled point cloud as text
    scaled_points = np.hstack([points_xyz_scaled, points_rgb, points_error])
    np.savetxt(
        output_dir / "points3D_scaled.txt",
        scaled_points,
        fmt="%.6f %.6f %.6f %d %d %d %.6f",
        header="X Y Z R G B ERROR",
    )

    # Save scale info
    with open(output_dir / "scale_info.txt", "w") as f:
        f.write(f"Scale factor: {scale}\n")

    return scaled_images, points_xyz_scaled


def align_colmap_to_metric_scale(
    colmap_dir,
    image_dir,
    output_dir=None,
    num_images=None,
    debug=True,
    downsample_factor=1.0,
):
    """Main function to align COLMAP reconstruction to metric scale.

    Args:
        colmap_dir: Path to COLMAP sparse reconstruction directory
        image_dir: Path to images directory
        output_dir: Output directory for scaled reconstruction
        num_images: Limit number of images to process (for testing)
        debug: Enable debug visualizations
        downsample_factor: Factor by which images are downsampled (e.g., 4.0 for images_4)
    """

    # Setup debug directory
    if debug and output_dir:
        debug_dir = Path(output_dir) / "debug_visualizations"
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug visualizations will be saved to: {debug_dir}")
        if downsample_factor != 1.0:
            logger.info(f"Using downsample factor: {downsample_factor}x")

    # Load COLMAP data
    logger.info("Loading COLMAP data...")

    points3D_path = os.path.join(colmap_dir, "points3D.bin")
    cameras_path = os.path.join(colmap_dir, "cameras.bin")
    images_path = os.path.join(colmap_dir, "images.bin")

    try:
        point_ids, points_xyz, points_rgb, points_error = read_colmap_points3D_binary(points3D_path)
        cameras = read_colmap_intrinsics_binary(cameras_path)
    except FileNotFoundError:
        new_points3D_path = points3D_path.replace("colmap", "nerfstudio/colmap")
        new_cameras_path = cameras_path.replace("colmap", "nerfstudio/colmap")
        point_ids, points_xyz, points_rgb, points_error = read_colmap_points3D_binary(new_points3D_path)
        cameras = read_colmap_intrinsics_binary(new_cameras_path)
    images = read_colmap_extrinsics_binary(images_path)

    if num_images is not None:
        images = images[:num_images]

    # Extract COLMAP depths
    logger.info(f"Extracting COLMAP depths from {len(images)} images...")
    correspondences = extract_colmap_depths(images, cameras, point_ids, points_xyz, image_dir)
    logger.info(f"Found {len(correspondences)} images with valid points")

    # Run Monodepth
    logger.info("\nRunning Monodepth...")
    monodepth_model = MoGeModelWrapper()
    correspondences = extract_monodepth_depths(correspondences, monodepth_model, downsample_factor=downsample_factor)

    # Debug: Visualize depth comparisons
    if debug and output_dir:
        logger.info("\nCreating depth comparison visualizations...")
        num_viz_samples = min(10, len([c for c in correspondences if c["depths_metric"] is not None]))
        visualize_depth_comparison(
            correspondences,
            debug_dir,
            num_samples=num_viz_samples,
            downsample_factor=downsample_factor,
        )

    # Solve for scale
    logger.info("\nSolving for scale factor (mask-weighted L1)...")
    scale, stats = solve_scale_factor(correspondences)

    logger.info("\nResults:")
    logger.info(f"  Scale factor: {scale:.6f}")
    logger.info(f"  RMSE: {stats['rmse']:.4f} (weighted: {stats['weighted_rmse']:.4f})")
    logger.info(f"  Mean error: {stats['mean_error']:.4f} (weighted: {stats['weighted_mean_error']:.4f})")
    logger.info(f"  Median error: {stats['median_error']:.4f}")
    logger.info(f"  Correspondences: {stats['num_correspondences']}")
    logger.info(f"  Mean mask: {stats['mean_mask']:.4f}")
    logger.info(f"  Median mask: {stats['median_mask']:.4f}")

    # Debug: Create correlation plots and error analysis
    if debug and output_dir:
        logger.info("\nCreating correlation and error analysis plots...")
        plot_depth_correlation(correspondences, scale, debug_dir / "depth_correlation.png")
        create_error_heatmap(correspondences, scale, debug_dir)

    # Debug: Run sanity checks
    if debug:
        sanity_checks(correspondences, scale, stats)

    return scale, stats, correspondences


if __name__ == "__main__":
    """
    python metric_alignment.py \
        --dataset_dir /path/to/DL3DV-ALL-960P/1K \
        --scene_id <scene_id>
    """
    parser = argparse.ArgumentParser(description="Align COLMAP reconstruction to metric scale")

    # Option 1: Use dataset_dir + scene_id
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Path to dataset root directory (e.g., DL3DV-ALL-960P)",
    )
    parser.add_argument("--scene_id", default=None, help="Scene ID/name within dataset")

    # Option 2: Use explicit paths (overrides dataset_dir/scene_id if provided)
    parser.add_argument(
        "--colmap_dir",
        default=None,
        help="Path to COLMAP sparse/0 directory (overrides dataset_dir/scene_id)",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help="Path to images directory (overrides dataset_dir/scene_id)",
    )

    # Common arguments
    parser.add_argument("--output_dir", default=None, help="Output directory for scaled reconstruction")
    parser.add_argument("--num_images", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--no_debug", action="store_true", help="Disable debug visualizations")
    parser.add_argument(
        "--downsample_factor",
        type=float,
        default=4.0,
        help="Factor by which images are downsampled (e.g., 4.0 for images_4 folder)",
    )

    args = parser.parse_args()

    # Construct paths from dataset_dir and scene_id if not explicitly provided
    if args.colmap_dir is None or args.image_dir is None:
        if args.dataset_dir is None or args.scene_id is None:
            parser.error("Either provide --dataset_dir and --scene_id, or provide both --colmap_dir and --image_dir")

        scene_path = os.path.join(args.dataset_dir, args.scene_id)

        if args.colmap_dir is None:
            args.colmap_dir = os.path.join(scene_path, "colmap", "sparse", "0")

        if args.image_dir is None:
            # Determine image directory based on downsample factor
            if args.downsample_factor == 1.0:
                image_folder = "images"
            else:
                image_folder = f"images_{int(args.downsample_factor)}"
            args.image_dir = os.path.join(scene_path, image_folder)

        # Set default output_dir if not provided
        if args.output_dir is None:
            args.output_dir = os.path.join("./debug", args.scene_id, "scaled")

    # Validate paths exist
    if not os.path.exists(args.colmap_dir):
        parser.error(f"COLMAP directory does not exist: {args.colmap_dir}")
    if not os.path.exists(args.image_dir):
        parser.error(f"Image directory does not exist: {args.image_dir}")

    scale, stats, correspondences = align_colmap_to_metric_scale(
        args.colmap_dir,
        args.image_dir,
        args.output_dir,
        args.num_images,
        debug=not args.no_debug,
        downsample_factor=args.downsample_factor,
    )

    with open(os.path.join(args.output_dir, "scale_info.txt"), "w") as f:
        f.write(f"Scale factor: {scale}\n")
