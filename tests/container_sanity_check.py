#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanity checks for the artifixer container."""

import importlib
import sys


def check(label, fn):
    try:
        result = fn()
        print(f"  [OK] {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


failures = 0

# ── Core framework ──
print("\n=== Core Framework ===")
if not check("torch", lambda: f"{__import__('torch').__version__} (CUDA {__import__('torch').version.cuda})"):
    failures += 1
if not check("torchvision", lambda: __import__("torchvision").__version__):
    failures += 1

# ── GPU availability ──
print("\n=== GPU ===")
import torch

if not check("CUDA available", lambda: torch.cuda.is_available()):
    failures += 1
if torch.cuda.is_available():
    if not check("GPU name", lambda: torch.cuda.get_device_name(0)):
        failures += 1
    if not check(
        "GPU compute capability",
        lambda: f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}",
    ):
        failures += 1

# ── FlashAttention ──
print("\n=== FlashAttention ===")
# flash_attn namespace exists (created by FA4's flash_attn.cute) but has no __version__
# when FA2 wheel is not separately installed — that's fine for H100/GB200 targets.
check("flash_attn namespace", lambda: (__import__("flash_attn"), "importable (FA4 provides flash_attn.cute)")[1])

# FA3 - check both the package and the interface module
if not check("flash_attn_3 package", lambda: __import__("flash_attn_3").__name__):
    failures += 1
if not check("flash_attn_interface (FA3 API)", lambda: (__import__("flash_attn_interface"), "importable")[1]):
    failures += 1

# FA4
fa4_ok = check("flash_attn.cute (FA4)", lambda: (importlib.import_module("flash_attn.cute"), "importable")[1])
if not fa4_ok:
    # FA4 might fail on non-Blackwell GPUs at import time - check if package is at least installed
    check("flash-attn-4 installed (pip)", lambda: (importlib.metadata.version("flash-attn-4")))

# PyTorch FA backends
try:
    from torch.nn.attention import list_flash_attention_impls

    check("PyTorch FA backends", lambda: list_flash_attention_impls())
except ImportError:
    check("PyTorch FA backends", lambda: "list_flash_attention_impls not available")

# ── HuggingFace Libraries ──
print("\n=== HuggingFace Libraries ===")
if not check("diffusers", lambda: __import__("diffusers").__version__):
    failures += 1
if not check("transformers", lambda: __import__("transformers").__version__):
    failures += 1
if not check("accelerate", lambda: __import__("accelerate").__version__):
    failures += 1

# Check diffusers FA detection
try:
    from diffusers.utils.import_utils import is_flash_attn_3_available, is_flash_attn_available

    check("diffusers sees FA2", lambda: is_flash_attn_available())
    check("diffusers sees FA3", lambda: is_flash_attn_3_available())
except ImportError as e:
    print(f"  [WARN] diffusers FA detection: {e}")

# ── Diffusers internals used by our code ──
print("\n=== Diffusers Internals (used by artifixer) ===")
if not check(
    "dispatch_attention_fn",
    lambda: (importlib.import_module("diffusers.models.attention_dispatch").dispatch_attention_fn.__name__),
):
    failures += 1
if not check(
    "WanTransformer3DModel", lambda: (getattr(importlib.import_module("diffusers"), "WanTransformer3DModel").__name__)
):
    failures += 1
if not check("AutoencoderKLWan", lambda: (getattr(importlib.import_module("diffusers"), "AutoencoderKLWan").__name__)):
    failures += 1
if not check(
    "prompt_clean", lambda: (importlib.import_module("diffusers.pipelines.wan.pipeline_wan").prompt_clean.__name__)
):
    failures += 1
if not check(
    "retrieve_latents",
    lambda: (importlib.import_module("diffusers.pipelines.wan.pipeline_wan_i2v").retrieve_latents.__name__),
):
    failures += 1
if not check("VideoProcessor", lambda: (importlib.import_module("diffusers.video_processor").VideoProcessor.__name__)):
    failures += 1
if not check(
    "FP32LayerNorm", lambda: (importlib.import_module("diffusers.models.normalization").FP32LayerNorm.__name__)
):
    failures += 1
if not check(
    "WanAttnProcessor2_0",
    lambda: (
        getattr(
            importlib.import_module("diffusers.models.transformers.transformer_wan"), "WanAttnProcessor2_0"
        ).__name__
    ),
):
    failures += 1

# ── Training dependencies ──
print("\n=== Training Dependencies ===")
for pkg in [
    "einops",
    "scipy",
    "wandb",
    "tqdm",
    "PIL",
    "matplotlib",
    "cv2",
    "yaml",
    "torchmetrics",
    "imageio",
    "h5py",
    "av",
    "torch_fidelity",
    "ftfy",
    "numpy",
]:
    if not check(pkg, lambda p=pkg: (importlib.import_module(p), "importable")[1]):
        failures += 1

# ── Thirdparty: 3dgrut ──
print("\n=== 3dgrut ===")
if not check("threedgrut", lambda: (importlib.import_module("threedgrut"), "importable")[1]):
    failures += 1
if not check(
    "threedgrut.datasets.camera_models",
    lambda: (importlib.import_module("threedgrut.datasets.camera_models"), "importable")[1],
):
    failures += 1

# ── Model training code imports ──
print("\n=== Model Training Code Imports ===")
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mt_path = os.path.join(repo_root, "model_training")
if os.path.exists(mt_path):
    for mod in [
        "model_training.utils.pose_utils",
        "model_training.utils.train_utils",
        "model_training.data.utils",
        "model_training.data.dataset_base",
        "model_training.net.prope",
        "model_training.schedulers.flow_match",
    ]:
        if not check(mod, lambda m=mod: (importlib.import_module(m), "importable")[1]):
            failures += 1
else:
    print(f"  [SKIP] model_training not found at {mt_path}")

# ── Summary ──
print(f"\n{'='*40}")
if failures:
    print(f"RESULT: {failures} check(s) FAILED")
    sys.exit(1)
else:
    print("RESULT: All checks passed!")
    sys.exit(0)
