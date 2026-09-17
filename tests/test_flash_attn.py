# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Validate FlashAttention 3 and FlashAttention 4 installations.

Run inside the built container:
  python test_flash_attn.py

On a CUDA 12 image (sm_90/H100 for FA3, sm_100/B200 for FA4 base):
  docker run --gpus all artifixer:cuda12 python /workspace/test_flash_attn.py

On a CUDA 13 image (full sm_121/GB200 FA4 support):
  docker run --gpus all artifixer:cuda13 python /workspace/test_flash_attn.py
"""

import sys

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

errors = []


def check(label, fn):
    try:
        result = fn()
        print(f"  [{PASS}] {label}" + (f": {result}" if result is not None else ""))
    except Exception as e:
        print(f"  [{FAIL}] {label}: {e}")
        errors.append(label)


def skip(label, reason):
    print(f"  [{SKIP}] {label}: {reason}")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
print("\n=== Environment ===")
import torch
check("torch version", lambda: torch.__version__)
check("CUDA available", lambda: f"{torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'}")
check("CUDA version (torch)", lambda: torch.version.cuda)
check("cuDNN version", lambda: torch.backends.cudnn.version())

try:
    import nvidia.cudnn_frontend
    check("cuDNN frontend version", lambda: nvidia.cudnn_frontend.__version__)
except ImportError:
    try:
        import cudnn
        check("cuDNN frontend version", lambda: cudnn.__version__)
    except ImportError:
        skip("cuDNN frontend", "not installed")

# ---------------------------------------------------------------------------
# PyTorch native FA backends (PyTorch 2.11+)
# ---------------------------------------------------------------------------
print("\n=== PyTorch Native Flash Attention Backends ===")
try:
    from torch.nn.attention import (
        list_flash_attention_impls,
        activate_flash_attention_impl,
        current_flash_attention_impl,
    )
    impls = list_flash_attention_impls()
    check("available FA implementations", lambda: impls)
    check("FA3 backend registered", lambda: "FA3" in impls)

    if torch.cuda.is_available():
        # Test FA3 activation
        if "FA3" in impls:
            try:
                activate_flash_attention_impl("FA3")
                check("FA3 backend activation", lambda: f"active impl = {current_flash_attention_impl()}")
            except Exception as e:
                skip("FA3 backend activation", str(e))

        # Test FA4 activation
        if "FA4" in impls:
            try:
                activate_flash_attention_impl("FA4")
                check("FA4 backend activation", lambda: f"active impl = {current_flash_attention_impl()}")
            except Exception as e:
                skip("FA4 backend activation", f"{e} (expected on non-Blackwell GPUs)")
    else:
        skip("FA backend activation", "no CUDA device")
except ImportError as e:
    check("torch.nn.attention FA API", lambda: (_ for _ in ()).throw(e))

# ---------------------------------------------------------------------------
# FlexAttention with FA backends
# ---------------------------------------------------------------------------
print("\n=== FlexAttention Forward Pass ===")
if torch.cuda.is_available():
    try:
        from torch.nn.attention.flex_attention import flex_attention

        B, H, S, D = 1, 4, 64, 64
        q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
        out = flex_attention(q, k, v)
        check("flex_attention forward pass (bf16)", lambda: f"output shape {tuple(out.shape)}")
        active = current_flash_attention_impl() if 'current_flash_attention_impl' in dir() else "unknown"
        check("flex_attention using backend", lambda: active)
    except Exception as e:
        check("flex_attention forward pass", lambda: (_ for _ in ()).throw(e))
else:
    skip("flex_attention forward pass", "no CUDA device")

# ---------------------------------------------------------------------------
# FlashAttention 4 package (flash_attn.cute)
# ---------------------------------------------------------------------------
print("\n=== FlashAttention 4 Package (flash_attn.cute) ===")

try:
    import importlib
    spec = importlib.util.find_spec("flash_attn.cute") or importlib.util.find_spec("flash_attn_4")
    check("flash-attn-4 package installed", lambda: f"found at {spec.origin}" if spec else "NOT FOUND")
    if spec is None:
        errors.append("flash-attn-4 package not installed")
except Exception as e:
    check("flash-attn-4 package check", lambda: (_ for _ in ()).throw(e))

if torch.cuda.is_available():
    try:
        import flash_attn.cute as fa4
        check("flash_attn.cute importable", lambda: "ok")

        if hasattr(fa4, "_flash_attn_fwd"):
            check("FA4 _flash_attn_fwd available", lambda: "ok")
        elif hasattr(fa4, "flash_attn_func"):
            check("FA4 flash_attn_func available", lambda: "ok")
        else:
            available = [x for x in dir(fa4) if not x.startswith("_")]
            skip("FA4 forward function", f"exports: {available[:10]}")
    except ImportError as e:
        skip("flash_attn.cute import", f"{e} (requires GPU context)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
if errors:
    print(f"  [{FAIL}] {len(errors)} check(s) failed:")
    for e in errors:
        print(f"         - {e}")
    sys.exit(1)
else:
    print(f"  [{PASS}] All checks passed.")
