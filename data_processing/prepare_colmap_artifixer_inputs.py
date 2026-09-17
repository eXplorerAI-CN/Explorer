#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a COLMAP scene for ArtiFixer inference.

This command turns a COLMAP scene into the dataset layout consumed by
`model_eval.run_inference`. It prepares the input files, trains and renders a
3DGUT reconstruction, estimates metric scale, and generates captions. It does
not run ArtiFixer inference.
"""

from __future__ import annotations

import argparse
import shutil
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from threedgrut.datasets.utils import (
    Camera,
    CAMERA_MODEL_NAMES,
    Image,
    qvec_to_so3,
    read_colmap_extrinsics_binary,
    read_colmap_intrinsics_binary,
)

from data_processing.camera_trajectories import (
    assert_target_only_trajectory,
    camera_intrinsics_for_frame,
    frame_with_camera_intrinsics,
    opencv_w2c_to_opengl_c2w,
    read_camera_trajectory,
    trajectory_frame_count,
    transforms_json,
    write_json,
)
from data_processing.captioning.generate_captions import generate_caption_hdf5
from data_processing.render_3dgrut_colmap import (
    render_3dgrut_colmap,
    render_outputs_complete,
    selected_indices_for_render,
)
from data_processing.sparse_recon.metric_alignment import align_colmap_to_metric_scale
from data_processing.threedgrut_training import DEFAULT_THREEDGRUT_CONFIG_DIR, train_3dgrut

DEFAULT_THREEDGRUT_CONFIG = "apps/colmap_3dgut_sparse_mcmc"
DEFAULT_TEXT_ENCODER_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
DEFAULT_CAPTIONING_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_CAPTION_FILENAME = "caption.h5"
DEFAULT_RECON_SUBDIR = "reconstruction"
DEFAULT_CAMERA_CONDITIONING_SCALE = 0.01
SUPPORTED_CAMERA_MODELS = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"}

# Canonical pose contract for prepared ArtiFixer/AF3D reconstructed_colmap splits.
# The writer converts source COLMAP OpenCV W2C poses into NeRFStudio/OpenGL C2W
# transform_matrix values; AF3D converts them back through the inverse helper and
# validates selected anchors against the original COLMAP images.bin.


@dataclass(frozen=True)
class ColmapScene:
    cameras: list[Camera]
    images: list[Image]


@dataclass(frozen=True)
class PreparedPaths:
    scene_id: str
    scene_root: Path
    threedgrut_input_dir: Path
    reconstruction_run_dir: Path
    render_output_root: Path
    render_checkpoint_dir: Path
    eval_dataset_root: Path
    eval_scene_dir: Path
    recon_results_dir: Path
    prompt_dir: Path
    split_path: Path
    selected_indices_path: Path
    scale_dir: Path


@dataclass(frozen=True)
class PreparedTrajectory:
    root: Path
    render_dataset_dir: Path
    transforms_path: Path
    render_trajectory_path: Path
    selected_indices_path: Path
    target_indices_path: Path
    render_checkpoint_dir: Path


def prepared_paths(scene_root: Path, scene_id: str, reconstruction_steps: int) -> PreparedPaths:
    threedgrut_input_dir = scene_root / "3dgrut_input" / scene_id
    recon_results_dir = scene_root / "recon_results"
    render_output_root = recon_results_dir / scene_id
    render_checkpoint_dir = render_output_root / DEFAULT_RECON_SUBDIR / scene_id / f"ours_{reconstruction_steps}"
    return PreparedPaths(
        scene_id=scene_id,
        scene_root=scene_root,
        threedgrut_input_dir=threedgrut_input_dir,
        reconstruction_run_dir=scene_root / "3dgrut_runs",
        render_output_root=render_output_root,
        render_checkpoint_dir=render_checkpoint_dir,
        eval_dataset_root=threedgrut_input_dir.parent,
        eval_scene_dir=threedgrut_input_dir,
        recon_results_dir=recon_results_dir,
        prompt_dir=scene_root / "captions",
        split_path=scene_root / "split.json",
        selected_indices_path=scene_root / "selected_indices.json",
        scale_dir=scene_root / "metric_alignment",
    )


def resolve_colmap_paths(colmap_dir: Path) -> tuple[Path, Path]:
    colmap_dir = colmap_dir.resolve()
    image_dir = colmap_dir / "images"
    sparse_dir = colmap_dir / "sparse" / "0"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing COLMAP images directory: {image_dir}")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse_dir / name).is_file():
            raise FileNotFoundError(f"Missing COLMAP sparse file: {sparse_dir / name}")
    return image_dir, sparse_dir


def read_colmap_scene(sparse_dir: Path) -> ColmapScene:
    cameras = read_colmap_intrinsics_binary(sparse_dir / "cameras.bin")
    return ColmapScene(
        cameras=[cameras[camera_id] for camera_id in sorted(cameras)],
        images=read_colmap_extrinsics_binary(sparse_dir / "images.bin"),
    )


def scene_relative_path(paths: PreparedPaths, path: Path) -> str:
    scene_root = paths.scene_root.resolve()
    path = path.resolve()
    try:
        return path.relative_to(scene_root).as_posix()
    except ValueError:
        return str(path)


def read_selected_image_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_selected_indices(args: argparse.Namespace, images: Sequence[Image]) -> list[int]:
    if args.selected_image_names_file is not None:
        name_to_index = {Path(image.name).name: index for index, image in enumerate(images)}
        names = read_selected_image_names(args.selected_image_names_file)
        basenames = [Path(name).name for name in names]
        missing = [name for name in basenames if name not in name_to_index]
        assert not missing, f"Selected image names are not in the COLMAP model: {missing}"
        selected = [name_to_index[name] for name in basenames]
    else:
        selected = list(range(len(images)))

    assert selected, "At least one training view is required"
    assert len(set(selected)) == len(selected), f"Selected indices contain duplicates: {selected}"
    assert min(selected) >= 0 and max(selected) < len(
        images
    ), f"Selected indices must be in [0, {len(images) - 1}], got {selected}"
    return selected


def require_unique_basenames(images: Sequence[Image]) -> None:
    basenames = [Path(image.name).name for image in images]
    assert len(set(basenames)) == len(basenames), "COLMAP image basenames must be unique"


def source_image_path(image_dir: Path, image: Image) -> Path:
    for candidate in (image_dir / image.name, image_dir / Path(image.name).name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find image for COLMAP entry {image.name!r} under {image_dir}")


def scaled_camera_params(camera: Camera, sx: float, sy: float) -> np.ndarray:
    params = camera.params.astype(np.float64).copy()
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        assert np.isclose(sx, sy), f"Camera {camera.id} requires uniform scaling for {camera.model}"
        params[0] *= sx
        params[1] *= sx
        params[2] *= sy
    elif camera.model in {"PINHOLE", "OPENCV"}:
        params[0] *= sx
        params[1] *= sy
        params[2] *= sx
        params[3] *= sy
    else:
        assert False, f"Unsupported camera model for image-resolution scaling: {camera.model}"
    return params


def scale_image_observations(image: Image, sx: float, sy: float) -> Image:
    xys = image.xys.astype(np.float64).copy()
    if len(xys):
        xys[:, 0] *= sx
        xys[:, 1] *= sy
    return image._replace(xys=xys)


def scale_colmap_scene_to_images(image_dir: Path, scene: ColmapScene) -> ColmapScene:
    camera_by_id = {camera.id: camera for camera in scene.cameras}
    image_info_by_camera: dict[int, tuple[int, int, str, Path]] = {}
    for image in scene.images:
        assert image.camera_id in camera_by_id, f"COLMAP image {image.name!r} references missing camera {image.camera_id}"
        image_path = source_image_path(image_dir, image)
        with PILImage.open(image_path) as pil_image:
            width, height = pil_image.size
        previous = image_info_by_camera.setdefault(image.camera_id, (width, height, image.name, image_path))
        assert previous[:2] == (width, height), (
            f"COLMAP camera {image.camera_id} is shared by images with different sizes: "
            f"{previous[0]}x{previous[1]} and {width}x{height}"
        )

    scaled_cameras = []
    scale_by_camera: dict[int, tuple[float, float]] = {}
    for camera in scene.cameras:
        image_info = image_info_by_camera.get(camera.id)
        if image_info is None:
            scaled_cameras.append(camera)
            scale_by_camera[camera.id] = (1.0, 1.0)
            continue
        width, height, image_name, image_path = image_info
        if (width, height) == (int(camera.width), int(camera.height)):
            scaled_cameras.append(camera)
            scale_by_camera[camera.id] = (1.0, 1.0)
            continue

        sx = width / float(camera.width)
        sy = height / float(camera.height)
        assert np.isclose(sx, sy), (
            f"COLMAP camera/image size mismatch for {image_name!r}: "
            f"camera {camera.id} is {int(camera.width)}x{int(camera.height)}, "
            f"but {image_path} is {width}x{height}"
        )
        params = scaled_camera_params(camera, sx, sy)
        print(
            f"Scaling COLMAP camera {camera.id} from {int(camera.width)}x{int(camera.height)} "
            f"to {width}x{height} to match input images",
            flush=True,
        )
        scaled_cameras.append(camera._replace(width=width, height=height, params=params))
        scale_by_camera[camera.id] = (sx, sy)

    scaled_images = [
        scale_image_observations(image, *scale_by_camera[image.camera_id])
        for image in scene.images
    ]
    return ColmapScene(cameras=scaled_cameras, images=scaled_images)


def write_colmap_cameras(path: Path, cameras: Sequence[Camera]) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            model = CAMERA_MODEL_NAMES[camera.model]
            fid.write(struct.pack("<iiQQ", camera.id, model.model_id, int(camera.width), int(camera.height)))
            fid.write(struct.pack("<" + "d" * len(camera.params), *camera.params))


def write_colmap_images(path: Path, images: Sequence[Image]) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image in images:
            fid.write(struct.pack("<idddddddi", image.id, *image.qvec, *image.tvec, image.camera_id))
            fid.write(image_basename(image).encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(image.xys)))
            for xy, point3d_id in zip(image.xys, image.point3D_ids):
                fid.write(struct.pack("<ddq", float(xy[0]), float(xy[1]), int(point3d_id)))


def image_basename(image: Image) -> str:
    return Path(image.name).name


def reset_prepared_scene(path: Path) -> None:
    if path.exists():
        assert path.is_dir() and not path.is_symlink(), f"Expected prepared scene directory, got {path}"
        shutil.rmtree(path)
    (path / "images").mkdir(parents=True)
    (path / "sparse/0").mkdir(parents=True)


def symlink_images(source_image_dir: Path, target_image_dir: Path, images: Sequence[Image]) -> None:
    for image in images:
        (target_image_dir / image_basename(image)).symlink_to(source_image_path(source_image_dir, image).resolve())


def write_sparse_model(source_sparse_dir: Path, target_sparse_dir: Path, scene: ColmapScene) -> None:
    write_colmap_cameras(target_sparse_dir / "cameras.bin", scene.cameras)
    write_colmap_images(target_sparse_dir / "images.bin", scene.images)
    (target_sparse_dir / "points3D.bin").symlink_to((source_sparse_dir / "points3D.bin").resolve())


def image_to_nerfstudio_transform(image: Image) -> list[list[float]]:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = qvec_to_so3(image.qvec)
    world_to_camera[:3, 3] = image.tvec
    return opencv_w2c_to_opengl_c2w(world_to_camera).tolist()


def camera_intrinsics(camera: Camera) -> dict[str, float | int]:
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        values = {"fl_x": f, "fl_y": f, "cx": cx, "cy": cy}
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
        values = {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy}
    elif camera.model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = camera.params
        values = {"fl_x": f, "fl_y": f, "cx": cx, "cy": cy, "k1": k1}
    elif camera.model == "RADIAL":
        f, cx, cy, k1, k2 = camera.params
        values = {"fl_x": f, "fl_y": f, "cx": cx, "cy": cy, "k1": k1, "k2": k2}
    elif camera.model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = camera.params
        values = {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "p1": p1, "p2": p2}
    else:
        assert False, f"Unsupported camera model for ArtiFixer transforms: {camera.model}"
    return {
        "camera_model": "OPENCV",
        "w": int(camera.width),
        "h": int(camera.height),
        **{key: float(value) for key, value in values.items()},
    }


def shared_camera_intrinsics(scene: ColmapScene) -> dict[str, float | int]:
    camera_by_id = {camera.id: camera for camera in scene.cameras}
    used_cameras = [camera_by_id[camera_id] for camera_id in sorted({image.camera_id for image in scene.images})]
    unsupported = sorted({camera.model for camera in used_cameras} - SUPPORTED_CAMERA_MODELS)
    assert not unsupported, f"Unsupported camera models for ArtiFixer transforms: {unsupported}"

    intrinsics = camera_intrinsics(used_cameras[0])
    for camera in used_cameras[1:]:
        assert camera_intrinsics(camera) == intrinsics, "ArtiFixer COLMAP prep expects one shared intrinsic calibration"
    return intrinsics


def source_frame(image: Image) -> dict[str, object]:
    return {
        "file_path": f"images/{image_basename(image)}",
        "transform_matrix": image_to_nerfstudio_transform(image),
    }


def write_transforms_json(path: Path, scene: ColmapScene) -> None:
    write_json(
        path,
        transforms_json(
            shared_camera_intrinsics(scene),
            [source_frame(image) for image in scene.images],
        ),
    )


def eval_split_entry(
    paths: PreparedPaths,
    *,
    transforms_path: Path,
    image_root: Path,
    render_dir: Path,
    opacity_dir: Path,
    selected_indices_path: Path,
    prompt_path: Path,
    metric_scale: float,
    reconstruction_checkpoint_path: Path,
    target_indices_path: Path | None = None,
    has_gt: bool = True,
) -> dict[str, object]:
    entry = {
        "scene_id": paths.scene_id,
        "transforms_path": scene_relative_path(paths, transforms_path),
        "image_root": scene_relative_path(paths, image_root),
        "render_dir": scene_relative_path(paths, render_dir),
        "opacity_dir": scene_relative_path(paths, opacity_dir),
        "selected_indices_path": scene_relative_path(paths, selected_indices_path),
        "prompt_path": scene_relative_path(paths, prompt_path),
        "reconstruction_checkpoint": scene_relative_path(paths, reconstruction_checkpoint_path),
        "metric_scale": metric_scale,
        "camera_scale": metric_scale * DEFAULT_CAMERA_CONDITIONING_SCALE,
    }
    if target_indices_path is not None:
        entry["target_indices_path"] = scene_relative_path(paths, target_indices_path)
    if not has_gt:
        entry["has_gt"] = False
    return entry


def trajectory_split_entry(
    paths: PreparedPaths,
    trajectory: PreparedTrajectory,
    metric_scale: float,
    reconstruction_checkpoint_path: Path,
) -> dict[str, object]:
    prompt_path = paths.prompt_dir / paths.scene_id / DEFAULT_CAPTION_FILENAME
    return eval_split_entry(
        paths,
        transforms_path=trajectory.transforms_path,
        image_root=paths.threedgrut_input_dir,
        render_dir=trajectory.render_checkpoint_dir / "renders",
        opacity_dir=trajectory.render_checkpoint_dir / "opacity",
        selected_indices_path=trajectory.render_checkpoint_dir / "selected_indices.json",
        prompt_path=prompt_path,
        metric_scale=metric_scale,
        reconstruction_checkpoint_path=reconstruction_checkpoint_path,
        target_indices_path=trajectory.target_indices_path,
        has_gt=False,
    )


def render_checkpoint_dir(paths: PreparedPaths, trajectory: PreparedTrajectory | None) -> Path:
    return paths.render_checkpoint_dir if trajectory is None else trajectory.render_checkpoint_dir


def render_frame_count(paths: PreparedPaths, trajectory: PreparedTrajectory | None) -> int:
    transforms_path = (
        paths.threedgrut_input_dir / "nerfstudio/transforms.json"
        if trajectory is None
        else trajectory.render_trajectory_path
    )
    return trajectory_frame_count(transforms_path)


def write_eval_split(
    paths: PreparedPaths,
    metric_scale: float,
    trajectory: PreparedTrajectory | None,
    reconstruction_checkpoint_path: Path,
) -> None:
    prompt_path = paths.prompt_dir / paths.scene_id / DEFAULT_CAPTION_FILENAME
    if trajectory is None:
        entry = eval_split_entry(
            paths,
            transforms_path=paths.threedgrut_input_dir / "nerfstudio/transforms.json",
            image_root=paths.threedgrut_input_dir,
            render_dir=paths.render_checkpoint_dir / "renders",
            opacity_dir=paths.render_checkpoint_dir / "opacity",
            selected_indices_path=paths.render_checkpoint_dir / "selected_indices.json",
            prompt_path=prompt_path,
            metric_scale=metric_scale,
            reconstruction_checkpoint_path=reconstruction_checkpoint_path,
        )
    else:
        entry = trajectory_split_entry(paths, trajectory, metric_scale, reconstruction_checkpoint_path)
    write_json(paths.split_path, {"test": {paths.scene_id: entry}})


def prepare_files(
    args: argparse.Namespace,
    paths: PreparedPaths,
    image_dir: Path,
    sparse_dir: Path,
    scene: ColmapScene,
    selected_indices: list[int],
) -> None:
    if paths.threedgrut_input_dir.exists() and not args.replace:
        print(f"Skipping prepare; found {paths.threedgrut_input_dir}", flush=True)
        return

    reset_prepared_scene(paths.threedgrut_input_dir)
    symlink_images(image_dir, paths.threedgrut_input_dir / "images", scene.images)
    write_sparse_model(sparse_dir, paths.threedgrut_input_dir / "sparse/0", scene)
    write_transforms_json(paths.threedgrut_input_dir / "nerfstudio/transforms.json", scene)
    write_selected_indices(paths, selected_indices, scene.images)


def write_selected_indices(paths: PreparedPaths, selected_indices: list[int], images: Sequence[Image]) -> None:
    selected_images = [image_basename(images[index]) for index in selected_indices]
    write_json(paths.selected_indices_path, selected_indices)
    (paths.scene_root / "selected_images.txt").write_text("\n".join(selected_images) + "\n")


def reconstruction_output_dir(paths: PreparedPaths) -> Path:
    return paths.reconstruction_run_dir / paths.scene_id / paths.scene_id


def reconstruction_checkpoint(paths: PreparedPaths, args: argparse.Namespace) -> Path:
    if args.reconstruction_checkpoint is not None:
        return args.reconstruction_checkpoint
    return (
        reconstruction_output_dir(paths) / f"ours_{args.reconstruction_steps}" / f"ckpt_{args.reconstruction_steps}.pt"
    )


def require_reconstruction_checkpoint(paths: PreparedPaths, args: argparse.Namespace) -> Path:
    checkpoint = reconstruction_checkpoint(paths, args)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing 3DGUT checkpoint: {checkpoint}")
    return checkpoint


def threedgrut_train_overrides(paths: PreparedPaths, args: argparse.Namespace) -> list[str]:
    return [
        f"path={paths.threedgrut_input_dir}",
        f"out_dir={paths.reconstruction_run_dir}",
        f"selected_indices_file={paths.selected_indices_path}",
        "test_last=False",
        "export_ingp.enabled=False",
        f"experiment_name={paths.scene_id}",
        f"n_iterations={args.reconstruction_steps}",
        f"checkpoint.iterations=[{args.reconstruction_steps}]",
    ]


def run_reconstruction(args: argparse.Namespace, paths: PreparedPaths) -> bool:
    checkpoint = reconstruction_checkpoint(paths, args)
    if args.reconstruction_checkpoint is not None:
        require_reconstruction_checkpoint(paths, args)
        print(f"Skipping reconstruction; using {checkpoint}", flush=True)
        return True
    if checkpoint.is_file() and not args.replace:
        print(f"Skipping reconstruction; found {checkpoint}", flush=True)
        return True
    train_3dgrut(
        DEFAULT_THREEDGRUT_CONFIG,
        threedgrut_train_overrides(paths, args),
        DEFAULT_THREEDGRUT_CONFIG_DIR,
    )
    require_reconstruction_checkpoint(paths, args)
    return False


def render_reconstruction(args: argparse.Namespace, paths: PreparedPaths, checkpoint_reused: bool) -> None:
    source_trajectory_path = paths.threedgrut_input_dir / "nerfstudio/transforms.json"
    checkpoint = require_reconstruction_checkpoint(paths, args)
    frame_count = render_frame_count(paths, None)
    if (
        not args.replace
        and checkpoint_reused
        and render_outputs_complete(
            paths.render_checkpoint_dir,
            frame_count,
            expected_selected_indices=selected_indices_for_render(paths.selected_indices_path, None),
        )
    ):
        print(f"Skipping render; found complete outputs in {paths.render_checkpoint_dir}", flush=True)
        return

    paths.render_checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    render_3dgrut_colmap(
        checkpoint=checkpoint,
        colmap_dir=paths.threedgrut_input_dir,
        output_root=paths.render_output_root,
        experiment_name=DEFAULT_RECON_SUBDIR,
        selected_indices=paths.selected_indices_path,
        trajectory_path=source_trajectory_path,
        trajectory_output_subdir="",
    )


def trajectory_paths(paths: PreparedPaths) -> PreparedTrajectory:
    root = paths.scene_root / "trajectory"
    return PreparedTrajectory(
        root=root,
        render_dataset_dir=root / "render_dataset",
        transforms_path=root / "transforms.json",
        render_trajectory_path=root / "trajectory.json",
        selected_indices_path=root / "selected_indices.json",
        target_indices_path=root / "target_indices.json",
        render_checkpoint_dir=paths.render_checkpoint_dir / "trajectory",
    )


def write_trajectory_render_dataset(trajectory: PreparedTrajectory, trajectory_data: dict[str, object]) -> None:
    first_frame = trajectory_data["frames"][0]
    write_json(
        trajectory.render_dataset_dir / "transforms.json",
        transforms_json(
            trajectory_data,
            [frame_with_camera_intrinsics(first_frame, camera_intrinsics_for_frame(trajectory_data, first_frame))],
        ),
    )


def write_trajectory_inputs(
    args: argparse.Namespace,
    paths: PreparedPaths,
    scene: ColmapScene,
    selected_indices: list[int],
) -> PreparedTrajectory:
    trajectory = trajectory_paths(paths)
    trajectory_data = read_camera_trajectory(args.trajectory_path)
    assert_target_only_trajectory(trajectory_data, str(args.trajectory_path))
    trajectory_frames = trajectory_data["frames"]

    target_frames = [
        frame_with_camera_intrinsics(frame, camera_intrinsics_for_frame(trajectory_data, frame))
        for frame in trajectory_frames
    ]
    selected_start = len(target_frames)
    source_intrinsics = shared_camera_intrinsics(scene)
    selected_frames = [
        frame_with_camera_intrinsics(source_frame(scene.images[index]), source_intrinsics) for index in selected_indices
    ]

    if trajectory.root.exists():
        assert (
            trajectory.root.is_dir() and not trajectory.root.is_symlink()
        ), f"Expected trajectory directory, got {trajectory.root}"
        shutil.rmtree(trajectory.root)
    trajectory.root.mkdir(parents=True)
    write_json(
        trajectory.transforms_path,
        transforms_json(trajectory_data, target_frames + selected_frames),
    )
    write_json(
        trajectory.render_trajectory_path,
        transforms_json(trajectory_data, target_frames),
    )
    write_trajectory_render_dataset(trajectory, trajectory_data)
    write_json(trajectory.selected_indices_path, list(range(selected_start, selected_start + len(selected_indices))))
    write_json(trajectory.target_indices_path, list(range(len(target_frames))))
    return trajectory


def render_trajectory(
    args: argparse.Namespace,
    paths: PreparedPaths,
    scene: ColmapScene,
    selected_indices: list[int],
) -> PreparedTrajectory:
    checkpoint = require_reconstruction_checkpoint(paths, args)
    trajectory = write_trajectory_inputs(args, paths, scene, selected_indices)
    render_3dgrut_colmap(
        checkpoint=checkpoint,
        colmap_dir=paths.threedgrut_input_dir,
        output_root=paths.render_output_root,
        experiment_name=DEFAULT_RECON_SUBDIR,
        selected_indices=trajectory.selected_indices_path,
        render_dataset_dir=trajectory.render_dataset_dir,
        trajectory_path=trajectory.render_trajectory_path,
    )
    return trajectory


def run_metric_alignment(args: argparse.Namespace, paths: PreparedPaths) -> None:
    scale_info = paths.scale_dir / "scale_info.txt"
    if args.metric_scale is not None:
        if scale_info.is_file() and not args.replace:
            existing_scale = read_metric_scale(scale_info)
            assert np.isclose(existing_scale, args.metric_scale), (
                f"Existing metric scale at {scale_info} is {existing_scale}, "
                f"but --metric_scale={args.metric_scale}. Re-run with --replace."
            )
            print(f"Skipping metric alignment; found {scale_info}", flush=True)
            return
        paths.scale_dir.mkdir(parents=True, exist_ok=True)
        scale_info.write_text(f"Scale factor: {args.metric_scale}\n")
        return

    if scale_info.is_file() and not args.replace:
        print(f"Skipping metric alignment; found {scale_info}", flush=True)
        return

    colmap_dir = args.colmap_dir / "sparse/0"
    image_dir = paths.eval_scene_dir / "images"
    paths.scale_dir.mkdir(parents=True, exist_ok=True)
    scale, _, _ = align_colmap_to_metric_scale(
        colmap_dir=colmap_dir,
        image_dir=image_dir,
        output_dir=paths.scale_dir,
        debug=False,
        downsample_factor=1,
    )
    scale_info.write_text(f"Scale factor: {scale}\n")


def generate_caption(args: argparse.Namespace, paths: PreparedPaths) -> None:
    caption_path = paths.prompt_dir / paths.scene_id / DEFAULT_CAPTION_FILENAME
    if caption_path.is_file() and not args.replace:
        print(f"Skipping captioning; found {caption_path}", flush=True)
        return

    generate_caption_hdf5(
        input_path=paths.eval_scene_dir,
        output_path=caption_path,
        dataset_downsample_factor=1,
        captioning_model_id=DEFAULT_CAPTIONING_MODEL_ID,
        text_encoder_model_id=args.text_encoder_model_id,
    )


def read_metric_scale(path: Path) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("Scale factor:"):
            return float(line.split(":", 1)[1].split()[0])
    assert False, f"Could not find scale factor in {path}"


def metric_scale(args: argparse.Namespace, paths: PreparedPaths) -> float | None:
    if args.metric_scale is not None:
        return args.metric_scale
    scale_info = paths.scale_dir / "scale_info.txt"
    return read_metric_scale(scale_info) if scale_info.is_file() else None


def parse_phases(value: str) -> set[str]:
    phases = {phase.strip() for phase in value.split(",") if phase.strip()}
    valid = {"prepare", "reconstruct", "render", "scale", "caption"}
    unknown = phases - valid
    assert not unknown, f"Unknown phases: {sorted(unknown)}"
    assert phases, f"--phases must include at least one of {sorted(valid)}"
    return phases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--colmap_dir",
        type=Path,
        required=True,
        help="Input COLMAP scene directory containing images/ and sparse/0/{cameras,images,points3D}.bin.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Scene output directory. The directory name is used as the scene id.",
    )
    parser.add_argument(
        "--selected_image_names_file",
        type=Path,
        default=None,
        help="Optional newline-delimited file of image filenames to use as 3DGRUT training views. "
        "Defaults to all COLMAP images.",
    )
    parser.add_argument(
        "--trajectory_path",
        type=Path,
        default=None,
        help="Optional transforms-style JSON camera trajectory to render and register for ArtiFixer inference. "
        "The JSON must include camera intrinsics and frames with 4x4 camera-to-world matrices.",
    )
    parser.add_argument(
        "--phases",
        default="prepare,reconstruct,render,scale,caption",
        help="Comma-separated phases to run. Valid phases: prepare,reconstruct,render,scale,caption.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        default=False,
        help="Overwrite existing prepared files, reconstruction renders, metric scale, and caption outputs.",
    )
    parser.add_argument(
        "--text_encoder_model_id",
        default=DEFAULT_TEXT_ENCODER_MODEL_ID,
        help="Wan text encoder model id used to store caption embeddings for ArtiFixer inference.",
    )
    parser.add_argument(
        "--reconstruction_steps",
        type=int,
        default=10000,
        help="3DGRUT MCMC training iterations and checkpoint step.",
    )
    parser.add_argument(
        "--reconstruction_checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional existing 3DGRUT checkpoint. When provided, reconstruction is skipped "
            "and this checkpoint is rendered."
        ),
    )
    parser.add_argument(
        "--metric_scale",
        type=float,
        default=None,
        help="Optional precomputed metric scale. When omitted, the scale phase runs MoGe alignment.",
    )

    return parser


def prepare_colmap_scene(args: argparse.Namespace) -> None:
    phases = parse_phases(args.phases)

    args.colmap_dir = args.colmap_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.selected_image_names_file is not None:
        args.selected_image_names_file = args.selected_image_names_file.expanduser().resolve()
    if args.reconstruction_checkpoint is not None:
        args.reconstruction_checkpoint = args.reconstruction_checkpoint.expanduser().resolve()
    if args.trajectory_path is not None:
        args.trajectory_path = args.trajectory_path.expanduser().resolve()
        assert "render" in phases, "--trajectory_path requires the render phase"
    image_dir, sparse_dir = resolve_colmap_paths(args.colmap_dir)
    scene = read_colmap_scene(sparse_dir)
    require_unique_basenames(scene.images)
    scene = scale_colmap_scene_to_images(image_dir, scene)

    selected_indices = resolve_selected_indices(args, scene.images)
    paths = prepared_paths(args.output_root, args.output_root.name, args.reconstruction_steps)
    paths.scene_root.mkdir(parents=True, exist_ok=True)

    if "prepare" in phases:
        prepare_files(args, paths, image_dir, sparse_dir, scene, selected_indices)
    if {"reconstruct", "render"} & phases:
        assert (
            paths.selected_indices_path.is_file()
        ), f"Missing prepared selected indices: {paths.selected_indices_path}"

    checkpoint_reused = True
    if "reconstruct" in phases:
        checkpoint_reused = run_reconstruction(args, paths)
    if "render" in phases and args.trajectory_path is None:
        render_reconstruction(args, paths, checkpoint_reused)
        trajectory = None
    elif "render" in phases:
        trajectory = render_trajectory(args, paths, scene, selected_indices)
    else:
        trajectory = None
    if "scale" in phases:
        run_metric_alignment(args, paths)
    if "caption" in phases:
        generate_caption(args, paths)
    scale = metric_scale(args, paths)
    render_dir = render_checkpoint_dir(paths, trajectory)
    if scale is not None and render_outputs_complete(render_dir, render_frame_count(paths, trajectory)):
        write_eval_split(paths, scale, trajectory, reconstruction_checkpoint(paths, args))

    print(f"prepared_scene={paths.scene_id}", flush=True)
    print(f"scene_root={paths.scene_root}", flush=True)
    print(f"selected_views={len(selected_indices)}", flush=True)
    print(f"split_path={paths.split_path}", flush=True)
    print(f"eval_dataset_root={paths.eval_dataset_root}", flush=True)
    print(f"prompt_dir={paths.prompt_dir}", flush=True)
    print(f"recon_results_dir={paths.recon_results_dir}", flush=True)
    print(f"reconstruction_checkpoint={reconstruction_checkpoint(paths, args)}", flush=True)
    if scale is not None:
        print(f"metric_scale={scale}", flush=True)
        print(f"camera_scale={scale * DEFAULT_CAMERA_CONDITIONING_SCALE}", flush=True)
    else:
        print("metric_scale=<missing>", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    prepare_colmap_scene(args)


if __name__ == "__main__":
    main()
