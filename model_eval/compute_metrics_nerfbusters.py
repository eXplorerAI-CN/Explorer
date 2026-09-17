# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compute metrics for Nerfbusters evaluation outputs.

Computes PSNR, SSIM, LPIPS, and FID for predicted vs ground truth images.
Only evaluates on TEST frames (frame_1_XXXXX.png naming convention).
"""

import argparse
import glob
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure as ssim_masked
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from data_processing.nerfbusters import NERFBUSTERS_SCENES, nerfbusters_gt_relpath
from model_eval.metrics_utils import FIDTracker, MetricsAggregator, compute_rgb_metrics, load_image_as_tensor


def get_train_test_split(transforms: dict) -> tuple[set, set]:
    """
    Get train/test split based on filename convention.

    Nerfbusters naming:
    - frame_XXXXX.png = TRAIN (no _1_ in middle)
    - frame_1_XXXXX.png = TEST (has _1_ in middle)
    """
    train_ids = set()
    test_ids = set()
    train_pattern = re.compile(r"^frame_\d+\.png$")

    for idx, frame in enumerate(transforms["frames"]):
        basename = os.path.basename(frame["file_path"])
        if train_pattern.match(basename):
            train_ids.add(idx)
        else:
            test_ids.add(idx)

    return train_ids, test_ids


def get_transforms_idx_to_mask_idx(transforms: dict, test_ids: set) -> dict:
    """
    Build mapping from transforms.json index to mask index.
    Masks are indexed by SORTED test filename order.
    """
    test_frames = []
    for idx in test_ids:
        basename = os.path.basename(transforms["frames"][idx]["file_path"])
        test_frames.append((idx, basename))

    test_frames.sort(key=lambda x: x[1])
    return {transforms_idx: mask_idx for mask_idx, (transforms_idx, _) in enumerate(test_frames)}


def load_mask(mask_path, target_size, device):
    """Load mask, resize to target size, and threshold at >= 1."""
    resample_nearest = getattr(Image, "Resampling", Image).NEAREST
    mask_image = Image.open(mask_path).convert("L").resize(target_size, resample_nearest)
    mask = torch.from_numpy(np.array(mask_image)).to(device)
    mask = (mask >= 1).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return mask


def resize_to_gt_if_needed(image: torch.Tensor, gt: torch.Tensor, *, label: str, frame_path: Path) -> torch.Tensor:
    if image.shape[-2:] == gt.shape[-2:]:
        return image
    warnings.warn(
        f"{label} frame {frame_path} has shape {tuple(image.shape[-2:])}, "
        f"expected GT shape {tuple(gt.shape[-2:])}; resizing for metrics.",
        RuntimeWarning,
        stacklevel=2,
    )
    return F.interpolate(image, size=gt.shape[-2:], mode="bicubic", align_corners=False)


def main(
    eval_output_name: str,
    num_views: int,
    *,
    eval_base_path: Path,
    nerfbusters_dir: Path,
    visibility_masks_dir: Path | None = None,
    use_masks: bool = True,
    sink_size: int = 7,
    output_suffix: str = "",
):
    if use_masks and visibility_masks_dir is None:
        raise ValueError("--visibility_masks_dir is required unless --no_masks is set")
    sink_suffix = f"_sink{sink_size}" if sink_size > 0 else ""
    suffix = f"_{output_suffix}" if output_suffix else ""
    eval_mode = f"nerfbusters_{num_views}_evenly_spaced{sink_suffix}{suffix}"
    path = eval_base_path / eval_output_name / eval_mode

    if not path.exists():
        raise FileNotFoundError(f"Eval path not found: {path}")

    device = torch.device("cuda")

    # Torchmetrics modules for masked metrics
    psnr_masked = PeakSignalNoiseRatio(data_range=1.0).to(device)
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

    processed_scene_count = 0

    with torch.inference_mode():
        for scene_id in NERFBUSTERS_SCENES:
            scene_path = path / scene_id
            if not scene_path.exists():
                print(f"Scene {scene_id} not found at {scene_path}")
                continue

            print(f"Processing scene {scene_id}...")

            # Load transforms to get train/test split and frame paths
            transforms_path = nerfbusters_dir / scene_id / "transforms.json"
            if not transforms_path.exists():
                print(f"  transforms.json not found for {scene_id}")
                continue

            with open(transforms_path, "r") as f:
                transforms = json.load(f)

            train_ids, test_ids = get_train_test_split(transforms)
            transforms_idx_to_mask_idx = get_transforms_idx_to_mask_idx(transforms, test_ids)

            # Build frame index to GT path mapping
            frame_idx_to_path = {}
            for i, frame in enumerate(transforms["frames"]):
                frame_idx_to_path[i] = nerfbusters_gt_relpath(scene_id, frame["file_path"])

            pred_dir = scene_path / "frames" / "batch_0000" / "pred"
            rendered_dir = scene_path / "frames" / "batch_0000" / "rendered"

            if not pred_dir.exists():
                print(f"  pred dir not found: {pred_dir}")
                continue

            pred_files = sorted(
                f for f in glob.glob(os.path.join(pred_dir, "*.png"))
                if not os.path.splitext(os.path.basename(f))[0].endswith("_mask")
            )
            rendered_files = sorted(
                f for f in glob.glob(os.path.join(rendered_dir, "*.png"))
                if not os.path.splitext(os.path.basename(f))[0].endswith("_mask")
            )
            assert len(pred_files) == len(
                rendered_files
            ), f"Mismatched file counts: {len(pred_files)} pred vs {len(rendered_files)} rendered in {scene_path}"

            if not pred_files:
                print(f"  No prediction files found in {pred_dir}")
                continue

            scene_test_count = 0
            for pred_file, rendered_file in zip(pred_files, rendered_files):
                # Predictions are named by transforms.json index
                frame_idx = int(os.path.splitext(os.path.basename(pred_file))[0])

                # Only evaluate on TEST frames
                if frame_idx not in test_ids:
                    continue

                if frame_idx not in frame_idx_to_path:
                    print(f"  Warning: frame {frame_idx} not in transforms")
                    continue

                gt_file_path = nerfbusters_dir / scene_id / frame_idx_to_path[frame_idx]
                if not gt_file_path.exists():
                    print(f"  Warning: GT not found: {gt_file_path}")
                    continue

                gt = load_image_as_tensor(gt_file_path, device)
                pred = load_image_as_tensor(pred_file, device)
                rendered = load_image_as_tensor(rendered_file, device)
                pred = resize_to_gt_if_needed(pred, gt, label="prediction", frame_path=Path(pred_file))
                rendered = resize_to_gt_if_needed(rendered, gt, label="3D-GUT", frame_path=Path(rendered_file))

                # Non-masked metrics using the shared RGB convention.
                ours_metrics.add(scene_id, **compute_rgb_metrics(pred, gt))
                gut_metrics.add(scene_id, **compute_rgb_metrics(rendered, gt))

                ours_fid.update(pred, gt)
                gut_fid.update(rendered, gt)

                # Masked metrics using torchmetrics
                # Masks indexed by sorted test filename order, predictions by transforms.json index
                if use_masks:
                    mask_idx = transforms_idx_to_mask_idx[frame_idx]
                    mask_path = visibility_masks_dir / scene_id / f"{mask_idx:05d}.png"
                    if mask_path.exists():
                        target_size = (gt.shape[3], gt.shape[2])  # (W, H)
                        mask = load_mask(mask_path, target_size, device)

                        pred_m = pred * mask
                        rendered_m = rendered * mask
                        gt_m = gt * mask

                        ours_metrics_masked.add(
                            scene_id,
                            psnr=psnr_masked(pred_m, gt_m).cpu().item(),
                            ssim=ssim_masked(pred_m, gt_m, data_range=1.0).cpu().item(),
                            lpips=lpips_masked(pred_m.clamp(0, 1), gt_m.clamp(0, 1)).detach().cpu().item(),
                        )

                        gut_metrics_masked.add(
                            scene_id,
                            psnr=psnr_masked(rendered_m, gt_m).cpu().item(),
                            ssim=ssim_masked(rendered_m, gt_m, data_range=1.0).cpu().item(),
                            lpips=lpips_masked(rendered_m.clamp(0, 1), gt_m.clamp(0, 1)).detach().cpu().item(),
                        )
                        ours_fid_masked.update(pred_m, gt_m)
                        gut_fid_masked.update(rendered_m, gt_m)

                scene_test_count += 1

            if scene_test_count > 0:
                processed_scene_count += 1
                gut_metrics.print_scene_summary(scene_id, prefix="3D-GUT ")
                ours_metrics.print_scene_summary(scene_id)
            else:
                print(f"  No test frames evaluated for {scene_id}")

        if processed_scene_count == 0:
            raise RuntimeError(f"No NerfBusters metrics computed under {path}")

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

        ours_metrics.save_to_yaml(path / "metrics.yaml", include_per_frame=True, fid=ours_fid_score)
        gut_metrics.save_to_yaml(path / "metrics_3dgut.yaml", include_per_frame=True, fid=gut_fid_score)

        if use_masks and ours_metrics_masked.get_all_values("psnr"):
            ours_metrics_masked.save_to_yaml(path / "metrics_masked.yaml", fid=ours_fid_masked_score)
            gut_metrics_masked.save_to_yaml(path / "metrics_3dgut_masked.yaml", fid=gut_fid_masked_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_output_name", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--sink_size", type=int, default=7, help="Sink size suffix (e.g., 7 for _sink7)")
    parser.add_argument("--no_masks", action="store_true")
    parser.add_argument("--eval_base_path", type=Path, required=True)
    parser.add_argument("--nerfbusters_dir", type=Path, required=True)
    parser.add_argument("--visibility_masks_dir", type=Path)
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()
    main(
        args.eval_output_name,
        args.num_views,
        eval_base_path=args.eval_base_path,
        nerfbusters_dir=args.nerfbusters_dir,
        visibility_masks_dir=args.visibility_masks_dir,
        use_masks=not args.no_masks,
        sink_size=args.sink_size,
        output_suffix=args.output_suffix,
    )
