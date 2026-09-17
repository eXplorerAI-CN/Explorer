# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import json
import numpy as np
import torch
import torchvision
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from scipy.spatial.transform import Rotation, Slerp

import threedgrut.datasets as datasets
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import create_summary_writer
from PIL import Image
import imageio

class Renderer:
    def __init__(
        self, model, conf, global_step, out_dir, path="", save_gt=True, writer=None, compute_extra_metrics=True, save_selected_indices=None
    ) -> None:

        if path:  # Replace the path to the test data
            conf.path = path

        self.model = model
        self.out_dir = out_dir
        self.save_gt = save_gt
        self.path = path
        self.conf = conf
        self.global_step = global_step
        self.dataset, self.dataloader = self.create_test_dataloader(conf)
        self.writer = writer
        self.compute_extra_metrics = compute_extra_metrics
        self.save_selected_indices = save_selected_indices if save_selected_indices is not None else self.dataset.selected_indices_file is not None

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

    def create_test_dataloader(self, conf):
        """Create the test dataloader for the given configuration."""
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        # test mode to ensure we render all the images regardless of the selected indices in ColmapDataset
        dataset = datasets.make_test(name=conf.dataset.type, config=conf)

        # Configure DataLoader arguments for the current platform
        dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": 8,
                "batch_size": 1,
                "shuffle": False,
                "collate_fn": None,
            }
        )

        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        return dataset, dataloader

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path, out_dir, path="", save_gt=True, writer=None, model=None, computes_extra_metrics=True,
        config_overrides=None
    ):
        """Loads checkpoint for test path.
        If path is stated, it will override the test path in checkpoint.
        If model is None, it will be loaded base on the
        """

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        global_step = checkpoint["global_step"]

        conf = checkpoint["config"]
        # overrides
        if conf["render"]["method"] == "3dgrt":
            conf["render"]["particle_kernel_density_clamping"] = True
            conf["render"]["min_transmittance"] = 0.03
        conf["render"]["enable_kernel_timings"] = True

        # Apply custom config overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if "." in key:
                    parts = key.split(".")
                    target = conf
                    for part in parts[:-1]:
                        target = target[part]
                    target[parts[-1]] = value
                else:
                    conf[key] = value

        object_name = Path(conf.path).stem
        if object_name == "nerfstudio": # DL3DV Benchmark dataset path always ends in nerfstudio parent directory name is more useful
            object_name = Path(conf.path).parent.stem
        experiment_name = conf["experiment_name"]
        writer, out_dir, run_name = create_summary_writer(conf, object_name, out_dir, experiment_name, use_wandb=False)

        if model is None:
            # Initialize the model and the optix context
            model = MixtureOfGaussians(conf)
            # Initialize the parameters from checkpoint
            model.init_from_checkpoint(checkpoint)
        model.build_acc()

        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=computes_extra_metrics,
        )

    @classmethod
    def from_preloaded_model(
        cls, model, out_dir, path="", save_gt=True, writer=None, global_step=None, compute_extra_metrics=False
    ):
        """Loads checkpoint for test path."""

        conf = model.conf
        if global_step is None:
            global_step = ""
        model.build_acc()
        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=compute_extra_metrics,
        )

    @torch.no_grad()
    def render_all(self):
        """Render all the images in the test dataset and log the metrics."""

        # Criterions that we log during training
        criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}

        if self.compute_extra_metrics:
            criterions |= {
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda"),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda"),
            }

        output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders")
        output_path_opacity = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "opacity")
        output_path_depth = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "depth")
        os.makedirs(output_path_renders, exist_ok=True)
        os.makedirs(output_path_opacity, exist_ok=True)
        os.makedirs(output_path_depth, exist_ok=True)

        if self.save_selected_indices:
            output_path_selected_indices = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "selected_indices.json")
            with open(self.dataset.selected_indices_file, "r") as f:
                selected_indices = json.load(f)
            indices = selected_indices[:self.dataset.num_selected_indices]

            with open(output_path_selected_indices, "w") as f:
                json.dump(indices, f)

        if self.save_gt:
            output_path_gt = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "gt")
            os.makedirs(output_path_gt, exist_ok=True)

        psnr = []
        ssim = []
        lpips = []
        inference_time = []
        test_images = []

        best_psnr = -1.0
        worst_psnr = 2**16 * 1.0

        best_psnr_img = None
        best_psnr_img_gt = None

        worst_psnr_img = None
        worst_psnr_img_gt = None

        logger.start_progress(task_name="Rendering", total_steps=len(self.dataloader), color="orange1")

        for iteration, batch in enumerate(self.dataloader):

            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)

            # Compute the outputs of a single batch
            outputs = self.model(gpu_batch)

            pred_rgb_full = outputs["pred_rgb"]
            pred_opacity_full = outputs["pred_opacity"]
            pred_dist_full = outputs["pred_dist"]
            rgb_gt_full = gpu_batch.rgb_gt

            # The values are already alpha composited with the background
            torchvision.utils.save_image(
                pred_rgb_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_renders, f"{iteration:05d}.png"),
            )
            Image.fromarray((pred_opacity_full * 255).round().byte().squeeze().detach().cpu().numpy()).save(
                os.path.join(output_path_opacity, f"{iteration:05d}.png")
            )
            np.save(
                os.path.join(output_path_depth, f"{iteration:05d}"),
                pred_dist_full.squeeze(0).detach().cpu().numpy(),
            )
            pred_img_to_write = pred_rgb_full[-1].clip(0, 1.0)
            gt_img_to_write = rgb_gt_full[-1].clip(0, 1.0)

            if self.writer is not None:
                test_images.append(pred_img_to_write)

            if self.save_gt:
                torchvision.utils.save_image(
                    rgb_gt_full.squeeze(0).permute(2, 0, 1),
                    os.path.join(output_path_gt, "{0:05d}".format(iteration) + ".png"),
                )

            # Compute the loss
            psnr_single_img = criterions["psnr"](outputs["pred_rgb"], gpu_batch.rgb_gt).item()
            psnr.append(psnr_single_img)  # evaluation on valid rays only
            logger.info(f"Frame {iteration}, PSNR: {psnr[-1]}")

            if psnr_single_img > best_psnr:
                best_psnr = psnr_single_img
                best_psnr_img = pred_img_to_write
                best_psnr_img_gt = gt_img_to_write

            if psnr_single_img < worst_psnr:
                worst_psnr = psnr_single_img
                worst_psnr_img = pred_img_to_write
                worst_psnr_img_gt = gt_img_to_write

            # evaluate on full image
            ssim.append(
                criterions["ssim"](
                    pred_rgb_full.permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )
            lpips.append(
                criterions["lpips"](
                    pred_rgb_full.clip(0, 1).permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )

            # Record the time
            inference_time.append(outputs["frame_time_ms"])

            logger.log_progress(task_name="Rendering", advance=1, iteration=f"{str(iteration)}", psnr=psnr[-1])

        logger.end_progress(task_name="Rendering")

        mean_psnr = np.mean(psnr)
        mean_ssim = np.mean(ssim)
        mean_lpips = np.mean(lpips)
        std_psnr = np.std(psnr)
        mean_inference_time = np.mean(inference_time)

        table = dict(
            mean_psnr=mean_psnr,
            mean_ssim=mean_ssim,
            mean_lpips=mean_lpips,
            std_psnr=std_psnr,
        )

        if self.conf.render.enable_kernel_timings:
            table["mean_inference_time"] = f"{'{:.2f}'.format(mean_inference_time)}" + " ms/frame"

        logger.log_table(f"⭐ Test Metrics - Step {self.global_step}", record=table)

        if self.writer is not None:
            self.writer.add_scalar("psnr/test", mean_psnr, self.global_step)
            self.writer.add_scalar("ssim/test", mean_ssim, self.global_step)
            self.writer.add_scalar("lpips/test", mean_lpips, self.global_step)
            self.writer.add_scalar("time/inference/test", mean_inference_time, self.global_step)

            if len(test_images) > 0:
                self.writer.add_images(
                    "image/pred/test",
                    torch.stack(test_images),
                    self.global_step,
                    dataformats="NHWC",
                )

            if best_psnr_img is not None:
                self.writer.add_images(
                    "image/best_psnr/test",
                    torch.stack([best_psnr_img, best_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

            if worst_psnr_img is not None:
                self.writer.add_images(
                    "image/worst_psnr/test",
                    torch.stack([worst_psnr_img, worst_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

        return mean_psnr, std_psnr, mean_inference_time

    @staticmethod
    def _frame_intrinsics(transforms_data: dict, frame: dict) -> dict:
        intrinsics = dict(transforms_data)
        for key in (
            "camera_model",
            "w",
            "h",
            "fl_x",
            "fl_y",
            "cx",
            "cy",
            "k1",
            "k2",
            "p1",
            "p2",
        ):
            if key in frame:
                intrinsics[key] = frame[key]
        return intrinsics

    def _step_output_dir(self, subdir: str = "") -> str:
        base = os.path.join(self.out_dir, f"ours_{int(self.global_step)}")
        return os.path.join(base, subdir) if subdir else base

    def _setup_output_dirs(self, subdir: str = "") -> tuple[str, str, str]:
        base = self._step_output_dir(subdir)
        paths = (
            os.path.join(base, "renders"),
            os.path.join(base, "opacity"),
            os.path.join(base, "depth"),
        )
        for path in paths:
            os.makedirs(path, exist_ok=True)
        return paths

    def _make_template_batch(self, camera_only: bool = False) -> dict:
        if not camera_only:
            template = self.dataset[0]
            batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in template.items()}
            batch["intr"] = torch.IntTensor([template["intr"]])
            return batch

        # transforms-schema trajectories can be camera-only, so omit the RGB target.
        intr = next(iter(self.dataset.intrinsics))
        return {
            "data": None,
            "pose": torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4),
            "intr": torch.IntTensor([intr]),
            "is_override": [False],
        }

    def _save_frame(self, outputs: dict, frame_idx: int, output_dirs: tuple[str, str, str], images: list, task_name: str) -> None:
        output_path_renders, output_path_opacity, output_path_depth = output_dirs
        pred_rgb_full = outputs["pred_rgb"]
        pred_opacity_full = outputs["pred_opacity"]
        pred_dist_full = outputs["pred_dist"]

        torchvision.utils.save_image(
            pred_rgb_full.squeeze(0).permute(2, 0, 1),
            os.path.join(output_path_renders, f"{frame_idx:05d}.png"),
        )
        Image.fromarray((pred_opacity_full * 255).round().byte().squeeze().detach().cpu().numpy()).save(
            os.path.join(output_path_opacity, f"{frame_idx:05d}.png")
        )
        np.save(
            os.path.join(output_path_depth, f"{frame_idx:05d}"),
            pred_dist_full.squeeze(0).detach().cpu().numpy(),
        )
        images.append((pred_rgb_full.squeeze(0) * 255).round().byte().detach().cpu().numpy())
        logger.log_progress(task_name=task_name, advance=1, iteration=str(frame_idx))

    @torch.no_grad()
    def render_trajectory(self, trajectory: list[int], interp_distance: float = 0.1):
        """
        Render a trajectory of camera poses with optional interpolation based on distance.

        Args:
            trajectory: List of dataset indices to render
            interp_distance: Distance in meters between interpolated frames. If 0, no interpolation is performed.
                           For example, 0.1 will generate a frame every 0.1 meters along the trajectory.
        """
        output_dirs = self._setup_output_dirs()
        step_dir = self._step_output_dir()
        task_name = "Rendering Trajectory"

        # Calculate total number of frames including interpolated ones
        total_frames = len(trajectory)
        if interp_distance > 0 and len(trajectory) > 1:
            for traj_idx in range(len(trajectory) - 1):
                batch = self.dataset[trajectory[traj_idx]]
                next_batch = self.dataset[trajectory[traj_idx + 1]]
                pose_current = batch["pose"].squeeze(0).cpu().numpy()
                pose_next = next_batch["pose"].squeeze(0).cpu().numpy()
                distance = np.linalg.norm(pose_next[:3, 3] - pose_current[:3, 3])
                total_frames += int(distance / interp_distance)

        logger.start_progress(task_name=task_name, total_steps=total_frames, color="orange1")

        images = []
        frame_idx = 0
        frame_to_gt_index = {}

        for traj_idx in range(len(trajectory)):
            frame_to_gt_index[frame_idx] = traj_idx
            batch = self.dataset[trajectory[traj_idx]]

            batch_for_render = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch_for_render["intr"] = torch.IntTensor([batch["intr"]])
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch_for_render)
            outputs = self.model(gpu_batch)
            outputs["pred_rgb"] = outputs["pred_rgb"].clamp(0, 1)
            self._save_frame(outputs, frame_idx, output_dirs, images, task_name)
            frame_idx += 1

            if traj_idx < len(trajectory) - 1 and interp_distance > 0:
                next_batch = self.dataset[trajectory[traj_idx + 1]]
                pose_current = batch["pose"].squeeze(0).cpu().numpy()
                pose_next = next_batch["pose"].squeeze(0).cpu().numpy()
                distance = np.linalg.norm(pose_next[:3, 3] - pose_current[:3, 3])
                num_interp_frames = int(distance / interp_distance)

                for interp_step in range(1, num_interp_frames + 1):
                    alpha = interp_step / (num_interp_frames + 1)
                    interp_pose = self.interpolate_single_pose(pose_current, pose_next, alpha)
                    interp_batch = {
                        "pose": torch.FloatTensor(interp_pose).view(batch_for_render["pose"].shape),
                        "intr": batch_for_render["intr"],
                    }
                    for key in ("data", "mask"):
                        if key in batch_for_render:
                            interp_batch[key] = batch_for_render[key]
                    gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(interp_batch)
                    outputs = self.model(gpu_batch)
                    self._save_frame(outputs, frame_idx, output_dirs, images, task_name)
                    frame_idx += 1

        imageio.mimsave(os.path.join(step_dir, "trajectory.mp4"), images, fps=15)
        with open(os.path.join(step_dir, "frame_to_gt_index.json"), "w") as f:
            json.dump(frame_to_gt_index, f, indent=4)

    @torch.no_grad()
    def render_from_file(self, trajectory_file: Path, output_subdir: str | None = None):
        # Callers can pass "" to keep renders directly under ours_STEP.
        subdir = trajectory_file.stem if output_subdir is None else output_subdir
        output_dirs = self._setup_output_dirs(subdir)
        step_dir = self._step_output_dir(subdir)
        task_name = "Rendering Trajectory"

        with trajectory_file.open() as f:
            trajectory_data = json.load(f)

        # Dict input is transforms.json-style; list input preserves the existing raw-pose path.
        uses_transforms_schema = isinstance(trajectory_data, dict)
        trajectory_intrinsics = None
        if uses_transforms_schema:
            assert hasattr(self.dataset, "transforms_pose_to_c2w"), (
                "transforms-schema trajectory rendering requires a dataset pose converter"
            )
            assert hasattr(self.dataset, "add_opencv_intrinsics_from_mapping"), (
                "transforms-schema trajectory rendering requires per-frame intrinsic registration"
            )
            frames = trajectory_data["frames"]
            assert isinstance(frames, list), f"{trajectory_file} frames must be a list"
            trajectory = []
            trajectory_intrinsics = []
            frame_sizes = set()
            for frame_idx, frame in enumerate(frames):
                assert isinstance(frame, dict), f"{trajectory_file} frame {frame_idx} must be a mapping"
                trajectory.append(frame["transform_matrix"])
                intrinsics = self._frame_intrinsics(trajectory_data, frame)
                frame_sizes.add((int(intrinsics["w"]), int(intrinsics["h"])))
                intr_id = -(frame_idx + 1)
                self.dataset.add_opencv_intrinsics_from_mapping(intr_id, intrinsics)
                trajectory_intrinsics.append(intr_id)
            assert (
                len(frame_sizes) == 1
            ), f"{trajectory_file} must use one fixed resolution, got {sorted(frame_sizes)}"
        else:
            trajectory = trajectory_data
        assert isinstance(trajectory, list), f"{trajectory_file} must contain a trajectory list or frames object"

        trajectory_poses = []
        for pose in trajectory:
            pose_array = np.array(pose, dtype=np.float32)
            if uses_transforms_schema:
                trajectory_poses.append(self.dataset.transforms_pose_to_c2w(pose_array))
            elif pose_array.shape == (4, 4):
                trajectory_poses.append(pose_array)
            else:
                to_add = np.eye(4, dtype=np.float32)
                to_add[:3] = pose_array
                trajectory_poses.append(to_add)

        logger.start_progress(task_name=task_name, total_steps=len(trajectory_poses), color="orange1")

        images = []
        poses = []
        template_batch = self._make_template_batch(camera_only=uses_transforms_schema)

        for pose_idx, pose in enumerate(trajectory_poses):
            poses.append(pose.tolist())
            intr = (
                template_batch["intr"]
                if trajectory_intrinsics is None
                else torch.IntTensor([trajectory_intrinsics[pose_idx]])
            )
            interp_batch = {
                "pose": torch.FloatTensor(pose).view(template_batch["pose"].shape),
                "intr": intr,
                "is_override": [False],
            }
            for key in ("data", "mask"):
                if key in template_batch:
                    interp_batch[key] = template_batch[key]
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(interp_batch)
            outputs = self.model(gpu_batch)
            self._save_frame(outputs, pose_idx, output_dirs, images, task_name)

        with open(os.path.join(step_dir, "poses.json"), "w") as f:
            json.dump(poses, f, indent=4)
        imageio.mimsave(os.path.join(step_dir, "trajectory.mp4"), images, fps=15)
        logger.end_progress(task_name=task_name)
        print(f"Rendered {len(trajectory_poses)} frames in trajectory")
        print(f"Output saved to: {step_dir}")

    def estimate_center_of_interest(self, poses: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
        """
        Estimate the 3D point that all cameras are looking at by finding the point that minimizes
        the sum of squared distances to all camera forward rays.

        Args:
            poses: Array of shape (N, 4, 4) containing C2W poses
            threshold: Convergence threshold for optimization

        Returns:
            np.ndarray: 3D point (x, y, z) representing estimated center of interest
        """
        camera_positions = poses[:, :3, 3]

        # Camera forward direction in world coordinates (negative Z-axis of camera frame)
        # For C2W matrices, the camera forward is -R[:, 2] (negative third column)
        camera_forwards = -poses[:, :3, 2]

        # Initial guess: centroid of camera positions
        center = camera_positions.mean(axis=0)

        # Iterative refinement using least squares
        # Find point that minimizes distance to all viewing rays
        for _ in range(100):
            # Compute vectors from cameras to current center estimate
            to_center = center - camera_positions

            # Project onto camera forward directions to find closest points on rays
            projections = (to_center * camera_forwards).sum(axis=1, keepdims=True)
            closest_points = camera_positions + projections * camera_forwards

            # New center is mean of closest points
            new_center = closest_points.mean(axis=0)

            # Check convergence
            if np.linalg.norm(new_center - center) < threshold:
                break

            center = new_center

        return center

    def compute_pose_distance(self, pose1: np.ndarray, pose2: np.ndarray) -> tuple[float, float]:
        """
        Compute the distance between two camera poses.

        Args:
            pose1: 4x4 C2W transformation matrix
            pose2: 4x4 C2W transformation matrix

        Returns:
            Tuple[float, float]: (translation_distance, rotation_distance)
                - translation_distance: Euclidean distance between camera positions
                - rotation_distance: Angular distance in radians between orientations
        """
        # Translation distance
        t1 = pose1[:3, 3]
        t2 = pose2[:3, 3]
        translation_dist = np.linalg.norm(t2 - t1)

        # Rotation distance (angular distance)
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]

        rot1 = Rotation.from_matrix(R1)
        rot2 = Rotation.from_matrix(R2)

        # Compute relative rotation
        relative_rotation = rot2 * rot1.inv()

        # Angular distance (magnitude of rotation axis-angle representation)
        rotation_dist = relative_rotation.magnitude()

        return translation_dist, rotation_dist

    def sort_poses_by_orbit_angle(self, camera_positions: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        Sort camera positions by their angle around the center of interest to create a proper orbit sequence.

        Uses PCA to find the principal plane of the orbit, then computes angles in that plane.

        Args:
            camera_positions: Array of shape (N, 3) containing camera positions
            center: 3D point representing center of interest

        Returns:
            np.ndarray: Array of indices that sort the poses in orbit order
        """
        # Center the positions
        centered_positions = camera_positions - center

        # Use PCA to find the principal plane of the orbit
        # The normal to this plane is the direction with least variance
        cov_matrix = np.cov(centered_positions.T)
        _, eigenvectors = np.linalg.eigh(cov_matrix)

        # Choose two orthogonal vectors in the orbit plane
        # Use the two eigenvectors with largest eigenvalues
        plane_u = eigenvectors[:, 2]
        plane_v = eigenvectors[:, 1]

        # Project camera positions onto the plane
        u_coords = (centered_positions @ plane_u)
        v_coords = (centered_positions @ plane_v)

        # Compute angles in the plane
        angles = np.arctan2(v_coords, u_coords)

        # Sort by angle
        sorted_indices = np.argsort(angles)

        return sorted_indices

    def interpolate_single_pose(self, pose1: np.ndarray, pose2: np.ndarray, alpha: float) -> np.ndarray:
        """
        Interpolate between two camera poses using SLERP for rotation and linear interpolation for translation.

        Args:
            pose1: 4x4 C2W transformation matrix (first pose)
            pose2: 4x4 C2W transformation matrix (second pose)
            alpha: Interpolation parameter (0 = pose1, 1 = pose2)

        Returns:
            np.ndarray: Interpolated 4x4 C2W transformation matrix
        """
        # Extract rotation and translation
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]
        t1 = pose1[:3, 3]
        t2 = pose2[:3, 3]

        # Convert rotation matrices to scipy Rotation objects
        rot1 = Rotation.from_matrix(R1)
        rot2 = Rotation.from_matrix(R2)

        # Use SLERP for smooth rotation interpolation
        key_times = [0, 1]
        key_rots = Rotation.concatenate([rot1, rot2])
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp([alpha])[0]

        # Linear interpolation for translation
        interp_t = (1 - alpha) * t1 + alpha * t2

        # Construct interpolated pose matrix
        interp_pose = np.eye(4)
        interp_pose[:3, :3] = interp_rot.as_matrix()
        interp_pose[:3, 3] = interp_t

        return interp_pose

    def interpolate_orbit_poses(self, poses: np.ndarray, loop: bool, interp_distance: float, rotation_weight: float = 1.0, training_poses: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate between camera poses following an orbit pattern around a common object of interest.

        Interpolation density is based on distance between poses rather than fixed frame count.
        This ensures consistent spacing along the trajectory.

        Args:
            poses: Either a numpy array of shape (N, 4, 4) or list of N 4x4 camera-to-world (C2W) transformation matrices
            interp_distance: Maximum distance between consecutive interpolated poses. Smaller values = denser interpolation.
                            Distance is computed as: sqrt(translation_dist^2 + (rotation_weight * rotation_dist)^2)
            loop: If True, interpolate from last pose back to first pose to complete the orbit
            center_of_interest: Optional 3D point (x, y, z) that cameras are looking at. If None, estimated automatically.
            rotation_weight: Weight for rotation distance relative to translation distance (default 1.0).
                            Higher values make rotation contribute more to total distance.
            first_training_pose: Optional 4x4 C2W matrix of a training pose to start the trajectory from.
                If provided, trajectory starts here before going through test poses.
        Returns:
            np.ndarray: Array of shape (M, 4, 4) containing interpolated C2W poses with approximately uniform spacing

        Example:
            >>> test_poses = np.array([...])  # Shape: (10, 4, 4)
            >>> # Interpolate with max 0.1 units between frames
            >>> orbit_trajectory = interpolate_orbit_poses(test_poses, interp_distance=0.1)
        """
        assert poses.shape[1:] == (4, 4), f"Expected poses of shape (N, 4, 4), got {poses.shape}"
        assert len(poses) >= 2, "Need at least 2 poses to interpolate"

        center_of_interest = self.estimate_center_of_interest(poses)

        # Combine training and test poses for orbit sorting
        if training_poses is not None:
            # Combine all poses
            all_poses = np.concatenate([training_poses, poses], axis=0)
            all_camera_positions = all_poses[:, :3, 3]

            # Sort all poses by orbit angle
            sorted_indices = self.sort_poses_by_orbit_angle(all_camera_positions, center_of_interest)

            # Find where the first training pose (index 0) is in the sorted order
            first_training_pose_position = np.where(sorted_indices == 0)[0][0]

            # Reorder to start from the first training pose
            sorted_indices = np.concatenate([
                sorted_indices[first_training_pose_position:],
                sorted_indices[:first_training_pose_position]
            ])

            sorted_poses = all_poses[sorted_indices]

            # Create a mapping: for each position in sorted order, what's its original index?
            # Training poses: indices 0 to len(training_poses)-1 -> map to -1 (not tracked)
            # Test poses: indices len(training_poses) to len(training_poses)+len(poses)-1 -> map to their test index
            num_training = len(training_poses)
            original_pose_indices = np.concatenate([
                np.full(num_training, -1, dtype=np.int32),  # Training poses get -1
                np.arange(len(poses), dtype=np.int32)        # Test poses get their index
            ])

            # Map sorted indices to original indices
            sorted_original_indices = original_pose_indices[sorted_indices]

            print(f"Starting orbit from first training pose")
            print(f"Incorporating {num_training} training poses and {len(poses)} test poses")
            print(f"Total orbit nodes: {len(sorted_poses)}")
        else:
            sorted_indices = self.sort_poses_by_orbit_angle(poses[:, :3, 3], center_of_interest)
            sorted_poses = poses[sorted_indices]
            sorted_original_indices = sorted_indices

        # Create pairs of poses to interpolate between
        pose_pairs = []
        pair_indices = []

        for i in range(len(sorted_poses) - 1):
            pose_pairs.append((sorted_poses[i], sorted_poses[i + 1]))
            pair_indices.append((sorted_original_indices[i], sorted_original_indices[i + 1]))

        # Optionally add loop back to start
        if loop:
            pose_pairs.append((sorted_poses[-1], sorted_poses[0]))
            pair_indices.append((sorted_original_indices[-1], sorted_original_indices[0]))

        # Interpolate between all pose pairs based on distance
        interpolated_poses = []
        original_indices = []

        for pair_idx, ((pose1, pose2), (idx1, idx2)) in enumerate(zip(pose_pairs, pair_indices)):
            # Compute distance between poses
            translation_dist, rotation_dist = self.compute_pose_distance(pose1, pose2)

            # Combined distance metric
            total_distance = np.sqrt(translation_dist**2 + (rotation_weight * rotation_dist)**2)

            # Determine number of interpolations needed
            num_interpolations = max(1, int(np.ceil(total_distance / interp_distance)))

            # Determine if this is the last pair (and not looping)
            is_last_pair = (pair_idx == len(pose_pairs) - 1) and not loop

            # Interpolate
            for j in range(num_interpolations):
                alpha = j / num_interpolations

                if alpha == 0.0:
                    # This is exactly the first pose
                    interp_pose = pose1
                    interpolated_poses.append(interp_pose)
                    original_indices.append(idx1)
                else:
                    # This is an interpolated pose
                    interp_pose = self.interpolate_single_pose(pose1, pose2, alpha)
                    interpolated_poses.append(interp_pose)
                    original_indices.append(-1)

            # Add the final pose if this is the last pair and we're not looping
            if is_last_pair:
                interpolated_poses.append(pose2)
                original_indices.append(idx2)

        return np.array(interpolated_poses), np.array(original_indices, dtype=np.int32)

    def render_orbit_trajectory(self, interp_distance_unnormalized: float, scale: float, training_pose_start):
        subdir = f"orbit_trajectory_{training_pose_start}_{interp_distance_unnormalized}"
        output_dirs = self._setup_output_dirs(subdir)
        step_dir = self._step_output_dir(subdir)
        task_name = "Rendering Orbit Trajectory"

        train_dataset = datasets.make(name=self.conf.dataset.type, config=self.conf, ray_jitter=None)[0]
        training_poses = train_dataset.poses
        training_poses = np.concatenate([
            training_poses[training_pose_start:training_pose_start + 1],
            training_poses[:training_pose_start],
            training_poses[training_pose_start + 1:],
        ])
        print(f"Incorporating {len(training_poses)} training poses into orbit trajectory")

        interpolated_poses, original_indices = self.interpolate_orbit_poses(
            self.dataset.poses, loop=False,
            interp_distance=interp_distance_unnormalized / scale,
            training_poses=training_poses,
        )

        logger.start_progress(task_name=task_name, total_steps=len(interpolated_poses), color="orange1")

        images = []
        frame_to_gt_index = {}
        poses = []
        template_batch = self._make_template_batch()

        for pose_idx in range(len(interpolated_poses)):
            if original_indices[pose_idx] != -1:
                frame_to_gt_index[pose_idx] = int(original_indices[pose_idx])
            poses.append(interpolated_poses[pose_idx].tolist())
            interp_batch = {
                "pose": torch.FloatTensor(interpolated_poses[pose_idx]).view(template_batch["pose"].shape),
                "intr": template_batch["intr"],
            }
            for key in ("data", "mask"):
                if key in template_batch:
                    interp_batch[key] = template_batch[key]
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(interp_batch)
            outputs = self.model(gpu_batch)
            self._save_frame(outputs, pose_idx, output_dirs, images, task_name)

        with open(os.path.join(step_dir, "poses.json"), "w") as f:
            json.dump(poses, f, indent=4)
        imageio.mimsave(os.path.join(step_dir, "orbit_trajectory.mp4"), images, fps=15)
        with open(os.path.join(step_dir, "frame_to_gt_index.json"), "w") as f:
            json.dump(frame_to_gt_index, f, indent=4)
