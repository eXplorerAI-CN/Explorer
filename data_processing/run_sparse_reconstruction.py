# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate ArtiFixer reconstruction HDF5 files."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from data_processing.scene_utils import (
    discover_scene_zips,
    read_scene_list,
    resolve_scene_path,
    scene_dl3dv_subdir,
    scene_name,
)
from data_processing.sparse_recon.convert_hdf5 import convert_reconstruction_to_hdf5
from data_processing.sparse_recon.half_covisibility_sampling import write_half_covisibility_sampled_indices
from data_processing.sparse_recon.metric_alignment import align_colmap_to_metric_scale
from data_processing.sparse_recon.metric_alignment_nerfstudio import align_nerfstudio_to_metric_scale
from data_processing.sparse_recon.workflow import (
    copy_reconstruction_outputs,
    extract_colmap_zip,
    extract_scene_zip,
    find_experiment_scene_dir,
    image_dir_for_scene,
    reconstruction_outputs,
    reconstruction_subdir_for_dl3dv_subdir,
    task_work_dir,
)
from data_processing.threedgrut_training import train_3dgrut


@dataclass(frozen=True)
class Scene:
    path: Path
    name: str
    dl3dv_subdir: str
    reconstruction_subdir: str


def resolve_scene(dl3dv_dir: Path, scene: str, reconstruction_subdir: str | None) -> Scene:
    scene_path = resolve_scene_path(dl3dv_dir, scene)
    return make_scene(dl3dv_dir, scene_path, reconstruction_subdir)


def make_scene(dl3dv_dir: Path, scene_path: Path, reconstruction_subdir: str | None) -> Scene:
    dl3dv_subdir = scene_dl3dv_subdir(dl3dv_dir, scene_path)
    recon_subdir = reconstruction_subdir or reconstruction_subdir_for_dl3dv_subdir(dl3dv_subdir)
    return Scene(scene_path, scene_name(scene_path), dl3dv_subdir, recon_subdir)


def require_threedgrut_root(args: argparse.Namespace) -> Path:
    root = args.threedgrut_root or args.repo_root / "thirdparty" / "3DGRUT-ArtiFixer"
    train_py = root / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(
            "ArtiFixer sparse reconstruction requires an ArtiFixer-compatible 3DGRUT checkout at "
            f"{root}. Clone the public fork into thirdparty/3DGRUT-ArtiFixer or pass --threedgrut_root."
        )
    return root


def prepare_scene_input(args: argparse.Namespace, scene: Scene, task_work_dir: Path) -> Path:
    if scene.path.suffix == ".zip":
        scene_dir = extract_scene_zip(scene.path, task_work_dir, scene.name)
        if args.colmap_zip_root is not None:
            colmap_zip = resolve_scene_path(args.colmap_zip_root, scene.name)
            extract_colmap_zip(colmap_zip, task_work_dir, scene_dir, scene.name)
        return scene_dir
    return scene.path


def find_colmap_dir(scene_dir: Path) -> Path | None:
    for rel in (Path("colmap/sparse/0"), Path("sparse/0")):
        candidate = scene_dir / rel
        if (candidate / "images.bin").exists():
            return candidate
    return None


def run_metric_alignment(args: argparse.Namespace, scene_input_dir: Path, experiment_scene_dir: Path) -> None:
    colmap_dir = find_colmap_dir(scene_input_dir)
    if colmap_dir is not None:
        scale, _, _ = align_colmap_to_metric_scale(
            colmap_dir=colmap_dir,
            image_dir=image_dir_for_scene(scene_input_dir, args.downsample_factor),
            output_dir=experiment_scene_dir,
            debug=False,
            downsample_factor=args.downsample_factor,
        )
        (experiment_scene_dir / "scale_info.txt").write_text(f"Scale factor: {scale}\n")
        return

    nerfstudio_dir = scene_input_dir / "nerfstudio"
    align_nerfstudio_to_metric_scale(
        scene_dir=nerfstudio_dir if nerfstudio_dir.exists() else scene_input_dir,
        output_dir=experiment_scene_dir,
        debug=False,
        downsample_factor=args.downsample_factor,
    )


def configure_3dgrut_environment(args: argparse.Namespace) -> None:
    if args.torch_cuda_arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = args.torch_cuda_arch_list
    os.environ.setdefault("CC", args.cc)
    os.environ.setdefault("CXX", args.cxx)
    slurm_job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else os.getpid()
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_{slurm_job_id}")


def train_sparse_reconstruction(
    args: argparse.Namespace,
    threedgrut_root: Path,
    train_path: Path,
    work_dir: Path,
    selected_indices_file: Path,
    views: int,
) -> None:
    configure_3dgrut_environment(args)
    train_3dgrut(
        args.sparse_recon_config,
        [
            f"path={train_path}/",
            f"out_dir={work_dir}/",
            f"selected_indices_file={selected_indices_file}",
            f"num_selected_indices={views}",
            f"dataset.downsample_factor={args.downsample_factor:g}",
            f"val_frequency={args.sparse_recon_val_frequency}",
            f"use_wandb={'True' if args.use_wandb else 'False'}",
            f"experiment_name={args.experiment_name}",
            f"n_iterations={args.n_steps}",
            f"checkpoint.iterations=[{args.n_steps}]",
        ],
        threedgrut_root / "configs",
    )


def run_one_reconstruction(args: argparse.Namespace, scene: Scene, scene_work_dir: Path, half: str, views: int) -> None:
    task_id = f"{scene.name}@{half}@{views}"
    outputs = reconstruction_outputs(args.output_root, scene.reconstruction_subdir, scene.name, half, views)
    if outputs.complete() and not args.replace_if_exists:
        print(f"Skipping {task_id}: final outputs already exist")
        return

    work_dir = task_work_dir(scene_work_dir, half, views)
    work_dir.mkdir(parents=True, exist_ok=True)
    scene_input_dir = prepare_scene_input(args, scene, work_dir)
    existing_dirs = {path.resolve() for path in work_dir.iterdir() if path.is_dir()}
    selected_indices_file = scene_work_dir / f"half_covisibility_sampled_indices_{half}.json"

    threedgrut_root = require_threedgrut_root(args)
    train_path = scene_input_dir / "nerfstudio" if args.nerfstudio_dir else scene_input_dir
    train_sparse_reconstruction(args, threedgrut_root, train_path, work_dir, selected_indices_file, views)

    experiment_scene_dir = find_experiment_scene_dir(
        work_dir,
        scene.name,
        expected_experiment_name=args.experiment_name,
        existing_dirs=existing_dirs,
    )
    run_metric_alignment(args, scene_input_dir, experiment_scene_dir)

    convert_reconstruction_to_hdf5(
        experiment_scene_dir,
        sparse_recon_n_steps=args.n_steps,
        crf=args.video_crf,
        codec=args.video_codec,
    )

    copy_reconstruction_outputs(experiment_scene_dir, outputs)
    print(f"Finished {task_id}")


def run_scene(args: argparse.Namespace, scene: Scene) -> None:
    scene_work_dir = args.work_root / scene.reconstruction_subdir / scene.name
    scene_work_dir.mkdir(parents=True, exist_ok=True)
    requested_outputs = [
        reconstruction_outputs(args.output_root, scene.reconstruction_subdir, scene.name, half, views)
        for half in args.scene_half
        for views in args.num_selected_indices
    ]
    if requested_outputs and all(outputs.complete() for outputs in requested_outputs) and not args.replace_if_exists:
        print(f"Skipping {scene.name}: all requested final outputs already exist")
        return

    missing_selected_indices = [
        half
        for half in args.scene_half
        if not (scene_work_dir / f"half_covisibility_sampled_indices_{half}.json").is_file()
    ]
    if args.replace_if_exists or missing_selected_indices:
        write_half_covisibility_sampled_indices(scene.path, scene_work_dir)
    else:
        print(f"Skipping half-covisibility for {scene.name}: selected-index files already exist")

    for half in args.scene_half:
        for views in args.num_selected_indices:
            run_one_reconstruction(args, scene, scene_work_dir, half, views)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dl3dv_dir", type=Path, required=True, help="Root containing DL3DV scene zips or dirs")
    parser.add_argument("--output_root", type=Path, required=True, help="Final reconstruction HDF5 root")
    parser.add_argument("--work_root", type=Path, required=True, help="Scratch/work root for 3DGRUT outputs")
    parser.add_argument("--scene_id", action="append", default=[], help="Only process this scene id/path")
    parser.add_argument("--scene_list", type=Path, help="Only process scenes listed in this file")
    parser.add_argument("--reconstruction_subdir", help="Override output subdir; default is dl3dv_<DL3DV subdir>")
    parser.add_argument("--num_selected_indices", type=int, nargs="+", required=True)
    parser.add_argument("--scene_half", nargs="+", choices=["0", "1"], default=["0", "1"])
    parser.add_argument("--colmap_zip_root", type=Path, help="Optional root containing <scene_id>.zip COLMAP exports")
    parser.add_argument("--nerfstudio_dir", action="store_true", help="Pass scene/nerfstudio as the 3DGRUT input path")
    parser.add_argument("--sparse_recon_config", default="apps/colmap_3dgut_sparse.yaml")
    parser.add_argument("--sparse_recon_val_frequency", type=int, default=999999)
    parser.add_argument("--experiment_name", default="artifixer_sparse_reconstruction")
    parser.add_argument("--n_steps", type=int, default=30000)
    parser.add_argument("--downsample_factor", type=float, default=4.0)
    parser.add_argument(
        "--replace_if_exists", action="store_true", help="Regenerate outputs even when final files exist"
    )
    parser.add_argument("--video_crf", type=int, default=0)
    parser.add_argument("--video_codec", choices=["libx264", "libx265"], default="libx264")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--torch_cuda_arch_list",
        help=(
            "Optional TORCH_CUDA_ARCH_LIST override for 3DGRUT extension builds. "
            "When omitted, any existing environment value is preserved and PyTorch otherwise detects visible GPUs."
        ),
    )
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--threedgrut_root", type=Path, help="Path to the ArtiFixer-compatible 3DGRUT checkout")
    args = parser.parse_args()

    scenes = list(args.scene_id)
    if args.scene_list is not None:
        scenes.extend(read_scene_list(args.scene_list))
    if scenes:
        args.scenes = [resolve_scene(args.dl3dv_dir, scene, args.reconstruction_subdir) for scene in scenes]
    else:
        scene_paths = discover_scene_zips(args.dl3dv_dir)
        if not scene_paths:
            parser.error(f"No scene zips found under {args.dl3dv_dir}")
        args.scenes = [make_scene(args.dl3dv_dir, scene_path, args.reconstruction_subdir) for scene_path in scene_paths]
    return args


def main() -> None:
    args = parse_args()
    require_threedgrut_root(args)
    for scene in args.scenes:
        run_scene(args, scene)


if __name__ == "__main__":
    main()
