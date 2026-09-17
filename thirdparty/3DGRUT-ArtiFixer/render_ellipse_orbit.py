# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render ellipse orbit trajectory through a 3dgrut model.

Generates ellipse camera poses from training views and renders them,
producing output compatible with the m360_orbit eval dataset format.

Usage:
    python render_ellipse_orbit.py \
        --checkpoint /path/to/ckpt_30000.pt \
        --out-dir /path/to/output \
        --scene garden \
        [--center 0.1,-0.2,0.3] \
        [--n-frames 81] \
        [--height auto]
"""

import argparse
import json
import os
from pathlib import Path

import imageio
import numpy as np
import torch
import torchvision
from PIL import Image

from threedgrut.render import Renderer


# ---------------------------------------------------------------------------
# Ellipse trajectory generation (adapted from gsplat datasets/traj.py)
# ---------------------------------------------------------------------------

def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def generate_ellipse_path_z(
    poses: np.ndarray,
    n_frames: int = 120,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = 0.0,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses.

    Args:
        center: Optional 3D lookat target override.
                If None, computed automatically via focus_point_fn.
    """
    if center is None:
        center = focus_point_fn(poses)

    offset = np.array([center[0], center[1], height])

    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    low = -sc + offset
    high = sc + offset
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                )
                + height,
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)
    positions = positions[:-1]

    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(center - p, up, p) for p in positions])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_poses(renderer, poses_4x4: np.ndarray, out_dir: str):
    """Render a list of 4x4 c2w poses through the 3dgrut model.

    Saves renders, opacity, and an mp4 video. Returns list of 3x4 poses
    (for saving as JSON).
    """
    dataset = renderer.dataset
    renders_dir = os.path.join(out_dir, "renders")
    opacity_dir = os.path.join(out_dir, "opacity")
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(opacity_dir, exist_ok=True)

    # Get a reference batch for shape/intrinsics
    ref_batch = dataset[0]
    ref_batch_expanded = {
        k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
        for k, v in ref_batch.items()
    }
    ref_batch_expanded["intr"] = torch.IntTensor([ref_batch["intr"]])

    images = []
    saved_poses = []

    for i, pose in enumerate(poses_4x4):
        batch = {
            "pose": torch.FloatTensor(pose).view(ref_batch_expanded["pose"].shape),
            "intr": ref_batch_expanded["intr"],
        }
        if "data" in ref_batch_expanded:
            batch["data"] = ref_batch_expanded["data"]
        if "mask" in ref_batch_expanded:
            batch["mask"] = ref_batch_expanded["mask"]

        gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
        outputs = renderer.model(gpu_batch)

        pred_rgb = outputs["pred_rgb"].clamp(0, 1)
        pred_opacity = outputs["pred_opacity"]

        torchvision.utils.save_image(
            pred_rgb.squeeze(0).permute(2, 0, 1),
            os.path.join(renders_dir, f"{i:05d}.png"),
        )
        Image.fromarray(
            (pred_opacity * 255).round().byte().squeeze().detach().cpu().numpy()
        ).save(os.path.join(opacity_dir, f"{i:05d}.png"))

        images.append(
            (pred_rgb.squeeze(0) * 255).round().byte().detach().cpu().numpy()
        )
        saved_poses.append(pose[:3, :4].tolist())

        if (i + 1) % 10 == 0 or i == len(poses_4x4) - 1:
            print(f"  Rendered {i + 1}/{len(poses_4x4)} frames")

    imageio.mimsave(os.path.join(out_dir, "rendered.mp4"), images, fps=15)
    return saved_poses


def main():
    parser = argparse.ArgumentParser(description="Render ellipse orbit through 3dgrut model")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to 3dgrut checkpoint (.pt)")
    parser.add_argument("--out-dir", required=True, type=str,
                        help="Output base directory. Renders saved to {out-dir}/{scene}/ellipse_{step}/")
    parser.add_argument("--scene", required=True, type=str,
                        help="Scene name (used for output directory naming)")
    parser.add_argument("--path", type=str, default="",
                        help="Override data path (if not baked into checkpoint)")
    parser.add_argument("--center", type=str, default=None,
                        help="Lookat center override as x,y,z (e.g. '0.1,-0.2,0.3')")
    parser.add_argument("--n-frames", type=int, default=81,
                        help="Number of orbit frames to render")
    parser.add_argument("--height", type=str, default="auto",
                        help="Camera height: 'auto' (mean z of training cameras), "
                             "'0' (origin), or a float value")
    parser.add_argument("--variation", type=float, default=0.0,
                        help="Height variation along orbit (0 = flat)")
    parser.add_argument("--phase", type=float, default=0.0,
                        help="Phase offset for height variation")
    args = parser.parse_args()

    # Load 3dgrut model
    print(f"Loading checkpoint: {args.checkpoint}")
    renderer = Renderer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        path=args.path,
        out_dir=args.out_dir,
        save_gt=False,
        computes_extra_metrics=False,
    )

    # Get training poses (4x4 c2w matrices)
    training_poses = renderer.dataset.get_poses()  # (N, 4, 4)
    print(f"Loaded {len(training_poses)} training poses")

    # Parse center override
    center = None
    if args.center is not None:
        center = np.array([float(x) for x in args.center.split(",")])
        assert center.shape == (3,), f"Center must be 3D, got shape {center.shape}"
        print(f"Using center override: {center}")
    else:
        auto_center = focus_point_fn(training_poses[:, :3, :])
        print(f"Using auto center (focus_point_fn): {auto_center}")

    # Compute height
    if args.height == "auto":
        height = training_poses[:, 2, 3].mean()
        print(f"Using auto height (mean z): {height:.4f}")
    else:
        height = float(args.height)
        print(f"Using height: {height}")

    # Generate ellipse path from 3x4 poses
    ellipse_poses_3x4 = generate_ellipse_path_z(
        training_poses[:, :3, :],
        n_frames=args.n_frames,
        variation=args.variation,
        phase=args.phase,
        height=height,
        center=center,
    )

    # Pad to 4x4
    bottom_row = np.array([[[0.0, 0.0, 0.0, 1.0]]]).repeat(len(ellipse_poses_3x4), axis=0)
    ellipse_poses_4x4 = np.concatenate([ellipse_poses_3x4, bottom_row], axis=1)

    # Determine output directory
    ckpt_step = int(renderer.global_step)
    scene_out_dir = os.path.join(args.out_dir, args.scene, f"ellipse_{ckpt_step}")
    print(f"Output: {scene_out_dir}")
    print(f"Rendering {args.n_frames} frames...")

    # Render
    saved_poses = render_poses(renderer, ellipse_poses_4x4, scene_out_dir)

    # Save poses JSON (3x4 matrices, matching m360_orbit.py expectations)
    poses_path = os.path.join(scene_out_dir, "ellipse_poses.json")
    with open(poses_path, "w") as f:
        json.dump(saved_poses, f, indent=4)

    print(f"Done. Saved {len(saved_poses)} frames to {scene_out_dir}")


if __name__ == "__main__":
    main()
