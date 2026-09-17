# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from zipfile import ZipFile

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

from data_processing.scene_utils import downsampled_image_path, load_scene_transforms_from_dir, load_scene_transforms_from_zip, scene_zip_member


def _scene_frame_root(dl3dv_dir: Path, scene_id: str) -> Path | None:
    scene_dir = dl3dv_dir / scene_id
    transforms_path = scene_dir / "transforms.json"
    if transforms_path.is_file():
        return scene_dir
    return None


def load_image_as_tensor(source: Union[Path, str, bytes, io.BytesIO], device: torch.device) -> torch.Tensor:
    """Load image and convert to (1, C, H, W) tensor normalized to [0, 1].

    Accepts a filesystem path or raw image bytes (for zip-sourced frames).
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with Image.open(source) as image:
        img = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0).to(device)


def _resolve_zip_root(zf: ZipFile, scene_id: str) -> str:
    """Return the scene-root member prefix used by the shared DL3DV zip loader."""
    _, scene_root = load_scene_transforms_from_zip(zf, scene_id)
    return scene_root


def load_dl3dv_scene_zip_bytes(
    dl3dv_dir: Path, subdir: str, scene_id: str, relative_paths: List[str]
) -> Dict[str, bytes]:
    """Read files from a DL3DV zip or extracted scene root, keyed by relative path."""
    frame_root = _scene_frame_root(dl3dv_dir, scene_id)
    if frame_root is not None:
        return {rel: (frame_root / rel).read_bytes() for rel in relative_paths}

    zip_path = dl3dv_dir / subdir / f"{scene_id}.zip"
    with ZipFile(zip_path, "r") as zf:
        scene_root = _resolve_zip_root(zf, scene_id)
        return {rel: zf.read(scene_zip_member(scene_root, rel)) for rel in relative_paths}


def load_dl3dv_scene_transforms(dl3dv_dir: Path, subdir: str, scene_id: str) -> dict:
    """Read transforms.json from a DL3DV zip or extracted scene root."""
    frame_root = _scene_frame_root(dl3dv_dir, scene_id)
    if frame_root is not None:
        return load_scene_transforms_from_dir(frame_root)

    zip_path = dl3dv_dir / subdir / f"{scene_id}.zip"
    with ZipFile(zip_path, "r") as zf:
        transforms, _ = load_scene_transforms_from_zip(zf, scene_id)
        return transforms


def load_subdir_by_scene_id(split_path: Path) -> Dict[str, str]:
    """Return a mapping of scene_id -> DL3DV subdir (e.g. "1K") derived from the split file."""
    with open(split_path) as f:
        split = json.load(f)
    mapping: Dict[str, str] = {}
    for bucket in ("test", "trainval"):
        for scene_id, recons in split[bucket].items():
            if recons:
                mapping[scene_id] = recons[0]["subdir"]
    return mapping


class MetricsAggregator:
    """Aggregates per-frame metrics across scenes."""

    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.scene_metrics: Dict[str, Dict[str, List[float]]] = {}

    def add(self, scene_id: str, **metrics):
        scene_metrics = self.scene_metrics.setdefault(scene_id, defaultdict(list))
        for name, value in metrics.items():
            scene_metrics[name].append(value)

    def print_scene_summary(self, scene_id: str, prefix: str = ""):
        metrics = self.scene_metrics.get(scene_id)
        if not metrics:
            return
        parts = [
            f"{name}: {np.mean(values):.4f}"
            for name in self.metric_names
            if (values := metrics.get(name))
        ]
        if parts:
            print(f"  {prefix}{', '.join(parts)}")

    def print_overall_summary(self, prefix: str = ""):
        for name in self.metric_names:
            all_vals = [v for scene in self.scene_metrics.values() for v in scene.get(name, [])]
            if all_vals:
                print(f"{prefix}{name}: {np.mean(all_vals):.4f}")

    def get_all_values(self, metric_name: str) -> List[float]:
        return [v for scene in self.scene_metrics.values() for v in scene.get(metric_name, [])]

    def to_results_dict(self, include_per_frame: bool = False) -> dict:
        results = {
            "scenes": {},
            "summary": {},
        }
        for scene_id, metrics in self.scene_metrics.items():
            scene_data = {}
            for name in self.metric_names:
                values = metrics.get(name)
                if values:
                    scene_data[f"{name}_mean"] = float(np.mean(values))
                    if include_per_frame:
                        scene_data[f"{name}_per_frame"] = values
            results["scenes"][scene_id] = scene_data

        for name in self.metric_names:
            all_vals = self.get_all_values(name)
            if all_vals:
                results["summary"][name] = float(np.mean(all_vals))

        return results

    def save_to_yaml(self, path: Path, include_per_frame: bool = False, **extra_metrics: Any):
        results = self.to_results_dict(include_per_frame)
        for key, value in extra_metrics.items():
            if value is not None:
                results["summary"][key] = float(value)
        with open(path, "w") as f:
            yaml.dump(results, f, default_flow_style=False)
        print(f"Results saved to {path}")


# GenFusion PSNR implementation
def psnr_genfusion(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """PSNR from GenFusion: https://github.com/Inception3D/GenFusion"""
    mse = ((img1 - img2) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def compute_rgb_metrics(image: torch.Tensor, target: torch.Tensor, lpips_net_type: str = "vgg") -> dict[str, float]:
    """Compute the shared unmasked RGB metrics used by eval scripts."""
    from model_eval.genfusion_lossutil import ssim
    from model_eval.lpipsPytorch import lpips

    return {
        "psnr": psnr_genfusion(image, target).item(),
        "ssim": ssim(image, target).item(),
        "lpips": lpips(image, target, net_type=lpips_net_type).item(),
    }


class FIDTracker:
    """Tracks and computes FID score across all images."""

    def __init__(self, device: torch.device, resize_to: int = 512):
        self.device = device
        self.resize = T.Resize((resize_to, resize_to))
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True, compute_on_cpu=True).to(device)
        self.pred_images: List[torch.Tensor] = []
        self.gt_images: List[torch.Tensor] = []

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """Add a pred/gt image pair (both [1, C, H, W] tensors)."""
        self.pred_images.append(self.resize(pred.detach().cpu()))
        self.gt_images.append(self.resize(gt.detach().cpu()))

    def compute(self, chunk_size: int = 64) -> Optional[float]:
        """Compute FID score. Returns None if no images were added."""
        if not self.pred_images:
            return None
        for i in range(0, len(self.pred_images), chunk_size):
            self.fid.update(torch.cat(self.pred_images[i : i + chunk_size]).to(self.device), real=False)
            self.fid.update(torch.cat(self.gt_images[i : i + chunk_size]).to(self.device), real=True)
        return self.fid.compute().item()
