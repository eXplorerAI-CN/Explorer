# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Render ellipse camera path for 3dgrut models using GenFusion's ellipse poses.
Transforms poses from GenFusion's normalized coordinate frame to 3dgrut's COLMAP frame.

Usage:
    python render_ellipse.py \
        --checkpoint /path/to/ckpt_30000.pt \
        --out-dir /path/to/output \
        --ellipse-poses /path/to/GenFusion/ellipse_poses.json \
        --colmap-dir /path/to/sparse/0
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

import threedgrut.datasets as datasets
from threedgrut.datasets.protocols import Batch
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger


# =============================================================================
# GenFusion normalization functions (to compute the transform we need to invert)
# =============================================================================

def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    """Get a similarity transform to normalize dataset from c2w cameras."""
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # Rotate the world so that z+ is the up axis
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # Recenter the scene
    if center_method == "focus":
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # Rescale the scene
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale

    return transform


def align_principle_axes(point_cloud):
    """Align point cloud to principal axes using PCA."""
    centroid = np.median(point_cloud, axis=0)
    translated_point_cloud = point_cloud - centroid
    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    rotation_matrix = eigenvectors.T
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid
    return transform


def transform_points(matrix, points):
    """Transform points using an SE(3) matrix."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix, camtoworlds):
    """Transform cameras using an SE(3) matrix."""
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


# =============================================================================
# COLMAP loading
# =============================================================================

def load_colmap_poses_and_points(colmap_dir: str):
    """Load poses and points from COLMAP sparse reconstruction."""
    from threedgrut.datasets.utils import qvec_to_so3, read_colmap_extrinsics_binary, read_next_bytes

    images_file = os.path.join(colmap_dir, "images.bin")
    cam_extrinsics = read_colmap_extrinsics_binary(images_file)

    poses = []
    for extr in cam_extrinsics:
        R = qvec_to_so3(extr.qvec)
        T = np.array(extr.tvec)
        W2C = np.zeros((4, 4), dtype=np.float32)
        W2C[:3, 3] = T
        W2C[:3, :3] = R
        W2C[3, 3] = 1.0
        C2W = np.linalg.inv(W2C)
        poses.append(C2W)
    poses = np.stack(poses, axis=0)

    points_file = os.path.join(colmap_dir, "points3D.bin")
    with open(points_file, "rb") as file:
        n_pts = read_next_bytes(file, 8, "Q")[0]
        points = np.zeros((n_pts, 3), dtype=np.float32)
        for i_pt in range(n_pts):
            pt_data = read_next_bytes(file, 43, "QdddBBBd")
            points[i_pt, :] = np.array(pt_data[1:4])
            t_len = read_next_bytes(file, num_bytes=8, format_char_sequence="Q")[0]
            read_next_bytes(file, num_bytes=8 * t_len, format_char_sequence="ii" * t_len)

    return poses, points


def compute_genfusion_transform(camtoworlds, points):
    """Compute the full GenFusion normalization transform (T2 @ T1)."""
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds_t1 = transform_cameras(T1, camtoworlds)
    points_t1 = transform_points(T1, points)
    T2 = align_principle_axes(points_t1)
    return T2 @ T1


def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


# =============================================================================
# Renderer
# =============================================================================

class EllipseRenderer:
    """Render ellipse path using GenFusion poses transformed to 3dgrut frame."""

    def __init__(self, checkpoint_path: str, out_dir: str, ellipse_poses_path: str, colmap_dir: str):
        self.out_dir = out_dir
        self.ellipse_poses_path = ellipse_poses_path
        self.colmap_dir = colmap_dir
        self.device = "cuda"

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.global_step = checkpoint["global_step"]
        self.conf = checkpoint["config"]

        if self.conf["render"]["method"] == "3dgrt":
            self.conf["render"]["particle_kernel_density_clamping"] = True
            self.conf["render"]["min_transmittance"] = 0.03
        self.conf["render"]["enable_kernel_timings"] = True

        # Initialize model
        self.model = MixtureOfGaussians(self.conf)
        self.model.init_from_checkpoint(checkpoint)
        self.model.build_acc()

        # Create dataset for intrinsics
        self.dataset = datasets.make_test(name=self.conf.dataset.type, config=self.conf)

        # Auto-detect colmap_dir if not provided
        if self.colmap_dir is None:
            data_path = self.conf.get("path", "")
            for subdir in ["sparse/0", "colmap/sparse/0"]:
                candidate = os.path.join(data_path, subdir)
                if os.path.exists(candidate):
                    self.colmap_dir = candidate
                    break
        logger.info(f"COLMAP directory: {self.colmap_dir}")

    def get_intrinsics(self):
        """Get camera intrinsics from the dataset."""
        for camera_id, (params_dict, rays_o, rays_d, camera_name) in self.dataset.intrinsics.items():
            fx = params_dict["focal_length"][0]
            fy = params_dict["focal_length"][1]
            resolution = params_dict["resolution"]
            width, height = int(resolution[0]), int(resolution[1])
            cx, cy = width / 2.0, height / 2.0
            return fx, fy, cx, cy, width, height
        raise ValueError("No intrinsics found in dataset")

    def transform_poses_to_3dgrut_frame(self, genfusion_poses: np.ndarray) -> np.ndarray:
        """Transform poses from GenFusion's normalized frame to 3dgrut's COLMAP frame."""
        if genfusion_poses.shape[1] == 3:
            genfusion_poses = pad_poses(genfusion_poses)

        # Compute GenFusion's normalization transform
        logger.info(f"Loading COLMAP data from {self.colmap_dir}")
        poses, points = load_colmap_poses_and_points(self.colmap_dir)
        logger.info(f"Computing transform from {len(poses)} poses and {len(points)} points")
        T = compute_genfusion_transform(poses, points)

        # Extract scale and compute inverse
        sR = T[:3, :3]
        st = T[:3, 3]
        scale = np.linalg.norm(sR[:, 0])
        R = sR / scale
        t = st / scale

        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T / scale
        T_inv[:3, 3] = -R.T @ t

        logger.info(f"Scale: {scale:.4f}, T_inv @ T check: {np.allclose(T_inv @ T, np.eye(4))}")

        # Transform each pose
        transformed_poses = []
        for pose in genfusion_poses:
            new_pose = T_inv @ pose
            U, _, Vt = np.linalg.svd(new_pose[:3, :3])
            new_pose[:3, :3] = U @ Vt
            transformed_poses.append(new_pose)

        return np.stack(transformed_poses, axis=0)

    @torch.no_grad()
    def render(self):
        """Render images along the transformed ellipse path."""
        fx, fy, cx, cy, width, height = self.get_intrinsics()
        logger.info(f"Intrinsics: fx={fx}, fy={fy}, size={width}x{height}")

        # Load and transform poses
        with open(self.ellipse_poses_path, "r") as f:
            genfusion_poses = np.array(json.load(f))
        logger.info(f"Loaded {len(genfusion_poses)} poses from {self.ellipse_poses_path}")

        ellipse_poses = self.transform_poses_to_3dgrut_frame(genfusion_poses)

        # Setup output
        output_path = Path(self.out_dir) / f"ellipse_{int(self.global_step)}"
        renders_path = output_path / "renders"
        opacity_path = output_path / "opacity"
        renders_path.mkdir(parents=True, exist_ok=True)
        opacity_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "ellipse_poses.json", "w") as f:
            json.dump(ellipse_poses[:, :3, :].tolist(), f, indent=4)

        # Render
        intrinsics = [float(fx), float(fy), float(cx), float(cy)]
        images = []

        from threedgrut.datasets.utils import pinhole_camera_rays
        u = np.tile(np.arange(width), height)
        v = np.arange(height).repeat(width)
        rays_o_cam, rays_d_cam = pinhole_camera_rays(u, v, fx, fy, width, height, None)
        rays_o = torch.tensor(rays_o_cam, dtype=torch.float32, device=self.device).reshape(1, height, width, 3)
        rays_d = torch.tensor(rays_d_cam, dtype=torch.float32, device=self.device).reshape(1, height, width, 3)

        logger.start_progress(task_name="Rendering", total_steps=len(ellipse_poses), color="cyan")
        for i, c2w in enumerate(ellipse_poses):
            pose = torch.tensor(c2w, dtype=torch.float32, device=self.device).unsqueeze(0)
            batch = Batch(rays_ori=rays_o, rays_dir=rays_d, T_to_world=pose, intrinsics=intrinsics)

            outputs = self.model(batch)
            pred_rgb = outputs["pred_rgb"]
            pred_opacity = outputs["pred_opacity"]

            torchvision.utils.save_image(pred_rgb.squeeze(0).permute(2, 0, 1), str(renders_path / f"{i:05d}.png"))
            Image.fromarray((pred_opacity * 255).round().byte().squeeze().detach().cpu().numpy()).save(
                str(opacity_path / f"{i:05d}.png")
            )
            images.append((pred_rgb.squeeze(0).cpu().numpy() * 255).astype(np.uint8))
            logger.log_progress(task_name="Rendering", advance=1)

        logger.end_progress(task_name="Rendering")

        video_path = output_path / "rendered.mp4"
        imageio.mimwrite(str(video_path), images, fps=15)
        logger.info(f"Saved video to {video_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Render 3dgrut using GenFusion's ellipse poses")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to 3dgrut checkpoint")
    parser.add_argument("--out-dir", required=True, type=str, help="Output directory")
    parser.add_argument("--ellipse-poses", required=True, type=str, help="Path to GenFusion's ellipse_poses.json")
    parser.add_argument("--colmap-dir", type=str, default=None, help="Path to COLMAP sparse/0 directory")
    args = parser.parse_args()

    renderer = EllipseRenderer(
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        ellipse_poses_path=args.ellipse_poses,
        colmap_dir=args.colmap_dir,
    )
    output_path = renderer.render()
    logger.info(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
