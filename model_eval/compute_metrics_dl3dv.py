# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared metrics computation for DL3DV test-scene eval, including ArtiFixer and DiFix variants."""

import argparse
import glob
import json
import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure as ssim_masked
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from model_eval.lpipsPytorch import lpips

from model_eval.dl3dv_difix_frame_remap import apply_difix_frame_remap
from model_eval.metrics_utils import (
    FIDTracker,
    MetricsAggregator,
    compute_rgb_metrics,
    downsampled_image_path,
    load_dl3dv_scene_transforms,
    load_dl3dv_scene_zip_bytes,
    load_image_as_tensor,
    load_subdir_by_scene_id,
)

DL3DV_EVALSETS = ("3dgrut_dl3dv_ours", "3dgrut_dl3dv_difix", "artifixer3d_dl3dv_ours", "artifixer3d_dl3dv_difix")
DIFIX_EVALSETS = {"3dgrut_dl3dv_difix", "artifixer3d_dl3dv_difix"}
OURS_EVALSETS = {"3dgrut_dl3dv_ours", "artifixer3d_dl3dv_ours"}


def _load_mask(mask_path: Path, target_size: tuple[int, int], device: torch.device, threshold: int = 1) -> torch.Tensor:
    """Load a visibility mask, resize to (W, H), and threshold count values."""
    resample_nearest = getattr(Image, "Resampling", Image).NEAREST
    mask_image = Image.open(mask_path).convert("L").resize(target_size, resample_nearest)
    mask = torch.from_numpy(np.array(mask_image)).to(device)
    return (mask >= threshold).float().unsqueeze(0).unsqueeze(0)


@lru_cache(maxsize=None)
def _load_split(split_path: Path) -> dict:
    with split_path.open() as f:
        return json.load(f)


def _hdf5_reconstruction_path(scene_id: str, num_views: int, split_path: Path) -> Path:
    split = _load_split(split_path)
    reconstructions = []
    for bucket in ("test", "trainval"):
        reconstructions.extend(split[bucket].get(scene_id, []))

    matches = [
        reconstruction
        for reconstruction in reconstructions
        if reconstruction["scene_half"] == "0" and reconstruction["num_selected_indices"] == num_views
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one DL3DV HDF5 path for scene_id={scene_id} num_views={num_views}, got {len(matches)}"
        )

    return Path(matches[0]["hdf5_path"])


def read_scene_id_list(path: Path) -> list[str]:
    scene_ids = []
    with path.open() as f:
        for line in f:
            scene_id = line.strip()
            if scene_id and not scene_id.startswith("#"):
                scene_ids.append(scene_id)
    return scene_ids


def load_metric_scene_ids(split_path: Path) -> list[str]:
    split = _load_split(split_path)
    return sorted(split["test"])


def load_hdf5_training_indices(scene_id: str, num_views: int, split_path: Path) -> set[int]:
    hdf5_file = _hdf5_reconstruction_path(scene_id, num_views, split_path)
    with h5py.File(hdf5_file, "r") as f:
        return set(f["selected_indices"][:].tolist())


def load_difix_training_indices(train_ids_dir: Path, scene_id: str) -> set[int]:
    """Load held-in DiFix training frame ids so metrics skip source-view frames."""
    with (train_ids_dir / scene_id / "difix_train_ids.json").open() as f:
        return set(json.load(f))


def build_difix_mask_index(transforms: dict, training_indices: set[int]) -> dict[int, int]:
    """Map transform frame ids to compact DiFix visibility-mask ids.

    DiFix visibility masks are numbered over sorted held-out/test frame names,
    while rendered predictions are named by transforms.json frame index.
    """
    test_frames = []
    for frame_idx, frame in enumerate(transforms["frames"]):
        if frame_idx in training_indices:
            continue
        test_frames.append((frame_idx, os.path.basename(frame["file_path"])))
    test_frames.sort(key=lambda item: item[1])
    return {frame_idx: mask_idx for mask_idx, (frame_idx, _) in enumerate(test_frames)}


def resolve_difix_mask_path(visibility_masks_dir: Path, scene_id: str, frame_idx: int, mask_idx: int | None) -> Path | None:
    scene_mask_dir = visibility_masks_dir / scene_id / "visibility_median_count"
    candidate_paths = []
    if mask_idx is not None:
        candidate_paths.append(scene_mask_dir / f"{mask_idx:05d}.png")
    candidate_paths.append(scene_mask_dir / f"{frame_idx:05d}.png")
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return None


def compute_metrics(
    eval_output_name: str,
    num_views: int,
    evalset: str,
    *,
    split_path: Path,
    dl3dv_dir: Path,
    eval_base_path: Path,
    use_masks: bool = True,
    sink_size: int = 7,
    render_trajectory: str = "val_frames",
    visibility_masks_dir: Path | None = None,
    test_scene_ids: Sequence[str] | None = None,
    training_indices_for_scene: Callable[[str], set[int]] | None = None,
    metrics_suffix: str = "",
    masked_lpips_backend: str = "torchmetrics",
) -> None:
    """Compute per-scene and aggregated metrics for rendered eval outputs.

    Args:
        eval_output_name: rendered eval output subdirectory under EVAL_BASE_PATH.
        num_views: number of training views used for evaluation.
        evalset: label used to construct the render output directory (e.g. "3dgrut_dl3dv_ours").
        use_masks: also compute visibility-masked variants of each metric.
        sink_size: optional attention-sink suffix encoded in the render dir name.
        split_path: split JSON used by eval and by ours HDF5 selected_indices lookup.
        dl3dv_dir: DL3DV zip root.
        eval_base_path: root containing rendered eval outputs.
        render_trajectory: trajectory suffix used by the renderer output directory.
        visibility_masks_dir: root containing optional visibility masks.
        test_scene_ids: explicit scene ids to evaluate for custom comparison paths.
        metrics_suffix: optional suffix for sidecar metric YAMLs.
        masked_lpips_backend: LPIPS implementation for masked metrics.
        training_indices_for_scene: optional callback for custom comparison eval paths.
            Defaults to reading selected_indices from the ArtiFixer HDF5 referenced by split_path.
    """
    if use_masks and visibility_masks_dir is None:
        raise ValueError("--visibility_masks_dir is required unless --no_masks is set")
    if test_scene_ids is None:
        test_scene_ids = load_metric_scene_ids(split_path)
    if training_indices_for_scene is None:
        training_indices_for_scene = lambda scene_id: load_hdf5_training_indices(scene_id, num_views, split_path)
    scene_id_to_subdir = load_subdir_by_scene_id(split_path)

    sink_suffix = f"_sink{sink_size}" if sink_size > 0 else ""
    trajectory_suffix = "" if render_trajectory == "val_frames" else f"_{render_trajectory}"
    eval_mode = f"distilled_views_{evalset}_{num_views}_evenly_spaced{sink_suffix}{trajectory_suffix}"
    path = eval_base_path / eval_output_name / eval_mode

    device = torch.device("cuda")

    psnr_masked = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips_masked = None
    if masked_lpips_backend == "torchmetrics":
        lpips_masked = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    metric_names = ["psnr", "ssim", "lpips"]
    ours_metrics = MetricsAggregator(metric_names)
    gut_metrics = MetricsAggregator(metric_names)
    ours_metrics_masked = MetricsAggregator(metric_names)
    gut_metrics_masked = MetricsAggregator(metric_names)
    ours_fid = FIDTracker(device)
    gut_fid = FIDTracker(device)
    ours_fid_masked = FIDTracker(device)
    gut_fid_masked = FIDTracker(device)

    def masked_lpips_value(image: torch.Tensor, target: torch.Tensor) -> float:
        image = image.clamp(0, 1)
        target = target.clamp(0, 1)
        if lpips_masked is not None:
            return lpips_masked(image, target).detach().cpu().item()
        return lpips(image, target, net_type=masked_lpips_backend).item()

    with torch.inference_mode():
        for scene_id in test_scene_ids:
            if not (path / scene_id).exists():
                print(f"Scene {scene_id} not found")
                continue
            print(f"Processing scene {scene_id}...")

            subdir = scene_id_to_subdir[scene_id]
            training_indices = training_indices_for_scene(scene_id)

            transforms = load_dl3dv_scene_transforms(dl3dv_dir, subdir, scene_id)
            if evalset in DIFIX_EVALSETS:
                transforms = apply_difix_frame_remap(transforms, scene_id)
            frame_idx_to_path = {
                i: downsampled_image_path(frame["file_path"], 4) for i, frame in enumerate(transforms["frames"])
            }
            difix_mask_index = build_difix_mask_index(transforms, training_indices) if evalset in DIFIX_EVALSETS else {}

            pred_dir = path / scene_id / "frames" / "batch_0000" / "pred"
            rendered_dir = path / scene_id / "frames" / "batch_0000" / "rendered"

            pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
            rendered_files = sorted(glob.glob(os.path.join(rendered_dir, "*.png")))
            if len(pred_files) != len(rendered_files):
                raise ValueError(
                    f"Mismatched file counts: {len(pred_files)} pred vs {len(rendered_files)} rendered for {scene_id}"
                )
            pred_names = [os.path.basename(path) for path in pred_files]
            rendered_names = [os.path.basename(path) for path in rendered_files]
            if pred_names != rendered_names:
                raise ValueError(f"Mismatched frame names between pred and rendered outputs for {scene_id}")

            if not pred_files:
                raise ValueError(f"No PNG frames found in {pred_dir} and {rendered_dir} for scene {scene_id}")
            frame_indices = [int(os.path.splitext(name)[0]) for name in pred_names]
            needed_relpaths = [
                frame_idx_to_path[frame_idx] for frame_idx in frame_indices if frame_idx not in training_indices
            ]
            gt_bytes_by_path = load_dl3dv_scene_zip_bytes(dl3dv_dir, subdir, scene_id, needed_relpaths)

            for pred_file, rendered_file, frame_idx in zip(pred_files, rendered_files, frame_indices):
                if frame_idx in training_indices:
                    continue

                gt = load_image_as_tensor(gt_bytes_by_path[frame_idx_to_path[frame_idx]], device)
                pred = load_image_as_tensor(pred_file, device)
                rendered = load_image_as_tensor(rendered_file, device)

                ours_metrics.add(scene_id, **compute_rgb_metrics(pred, gt))
                gut_metrics.add(scene_id, **compute_rgb_metrics(rendered, gt))
                ours_fid.update(pred, gt)
                gut_fid.update(rendered, gt)

                if use_masks:
                    mask_path = resolve_difix_mask_path(
                        visibility_masks_dir,
                        scene_id,
                        frame_idx,
                        difix_mask_index.get(frame_idx),
                    )
                    if mask_path is not None:
                        mask = _load_mask(mask_path, (gt.shape[3], gt.shape[2]), device, threshold=1)
                        pred_m, rendered_m, gt_m = pred * mask, rendered * mask, gt * mask

                        ours_metrics_masked.add(
                            scene_id,
                            psnr=psnr_masked(pred_m, gt_m).cpu().item(),
                            ssim=ssim_masked(pred_m, gt_m, data_range=1.0).cpu().item(),
                            lpips=masked_lpips_value(pred_m, gt_m),
                        )
                        gut_metrics_masked.add(
                            scene_id,
                            psnr=psnr_masked(rendered_m, gt_m).cpu().item(),
                            ssim=ssim_masked(rendered_m, gt_m, data_range=1.0).cpu().item(),
                            lpips=masked_lpips_value(rendered_m, gt_m),
                        )
                        ours_fid_masked.update(pred_m, gt_m)
                        gut_fid_masked.update(rendered_m, gt_m)

            gut_metrics.print_scene_summary(scene_id, prefix="3D-GUT ")
            ours_metrics.print_scene_summary(scene_id)
            if use_masks and ours_metrics_masked.scene_metrics.get(scene_id):
                gut_metrics_masked.print_scene_summary(scene_id, prefix="3D-GUT (masked) ")
                ours_metrics_masked.print_scene_summary(scene_id, prefix="(masked) ")

        ours_fid_score = ours_fid.compute()
        gut_fid_score = gut_fid.compute()
        ours_fid_masked_score = ours_fid_masked.compute() if use_masks else None
        gut_fid_masked_score = gut_fid_masked.compute() if use_masks else None

        print("\n=== Summary ===")
        gut_metrics.print_overall_summary(prefix="3D-GUT ")
        if gut_fid_score is not None:
            print(f"3D-GUT fid: {gut_fid_score:.4f}")
        ours_metrics.print_overall_summary()
        if ours_fid_score is not None:
            print(f"fid: {ours_fid_score:.4f}")

        if use_masks and ours_metrics_masked.get_all_values("psnr"):
            print()
            gut_metrics_masked.print_overall_summary(prefix="3D-GUT (masked) ")
            if gut_fid_masked_score is not None:
                print(f"3D-GUT (masked) fid: {gut_fid_masked_score:.4f}")
            ours_metrics_masked.print_overall_summary(prefix="(masked) ")
            if ours_fid_masked_score is not None:
                print(f"(masked) fid: {ours_fid_masked_score:.4f}")

        suffix = f"_{metrics_suffix}" if metrics_suffix else ""
        ours_metrics.save_to_yaml(path / f"metrics{suffix}.yaml", include_per_frame=True, fid=ours_fid_score)
        gut_metrics.save_to_yaml(path / f"metrics_3dgut{suffix}.yaml", include_per_frame=True, fid=gut_fid_score)

        if use_masks and ours_metrics_masked.get_all_values("psnr"):
            ours_metrics_masked.save_to_yaml(path / f"metrics_masked{suffix}.yaml", fid=ours_fid_masked_score)
            gut_metrics_masked.save_to_yaml(path / f"metrics_3dgut_masked{suffix}.yaml", fid=gut_fid_masked_score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_output_name", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--evalset", choices=DL3DV_EVALSETS, required=True)
    parser.add_argument("--sink_size", type=int, default=7, help="Sink size suffix (e.g. 7 for _sink7)")
    parser.add_argument("--render_trajectory", choices=["val_frames", "all_frames"], default="val_frames")
    parser.add_argument("--no_masks", action="store_true")
    parser.add_argument("--split_path", type=Path, required=True)
    parser.add_argument("--dl3dv_dir", type=Path, required=True)
    parser.add_argument("--eval_base_path", type=Path, required=True)
    parser.add_argument("--visibility_masks_dir", type=Path)
    parser.add_argument("--metrics_suffix", type=str, default="")
    parser.add_argument(
        "--masked_lpips_backend",
        choices=("torchmetrics", "alex", "squeeze", "vgg"),
        default="torchmetrics",
    )
    parser.add_argument(
        "--difix_train_ids_dir",
        type=Path,
        help="Directory containing per-scene difix_train_ids.json; required for DiFix evalsets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_scene_ids = None
    training_indices_for_scene = None
    if args.evalset in DIFIX_EVALSETS:
        if args.difix_train_ids_dir is None:
            raise ValueError("--difix_train_ids_dir is required for DiFix DL3DV evalsets")
        difix_scene_list_path = Path("data_processing/dl3dv_benchmark_difix_test_scenes.txt")
        test_scene_ids = read_scene_id_list(difix_scene_list_path)
        training_indices_for_scene = lambda scene_id: load_difix_training_indices(args.difix_train_ids_dir, scene_id)
    elif args.evalset in OURS_EVALSETS:
        ours_scene_list_path = Path("data_processing/dl3dv_benchmark_ours_test_scenes.txt")
        test_scene_ids = read_scene_id_list(ours_scene_list_path)

    compute_metrics(
        eval_output_name=args.eval_output_name,
        num_views=args.num_views,
        evalset=args.evalset,
        use_masks=not args.no_masks,
        sink_size=args.sink_size,
        render_trajectory=args.render_trajectory,
        split_path=args.split_path,
        dl3dv_dir=args.dl3dv_dir,
        eval_base_path=args.eval_base_path,
        visibility_masks_dir=args.visibility_masks_dir,
        test_scene_ids=test_scene_ids,
        training_indices_for_scene=training_indices_for_scene,
        metrics_suffix=args.metrics_suffix,
        masked_lpips_backend=args.masked_lpips_backend,
    )


if __name__ == "__main__":
    main()
