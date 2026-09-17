# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark all attention dispatch paths on the current GPU.

Usage:
  docker run --gpus all artifixer:cuda12 python /workspace/benchmark_attention.py
"""

import time
import torch
import torch.nn.functional as F
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
    list_flash_attention_impls,
    restore_flash_attention_impl,
)
from torch.nn.attention.flex_attention import flex_attention

flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
WARMUP = 10
ITERS = 50
SHAPES = [
    # (B, H, S, D) — representative of training workloads
    (1, 40, 2048, 128),
    (2, 40, 4096, 128),
]
DTYPE = torch.bfloat16

gpu_name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print(f"GPU: {gpu_name}  compute capability: {cc[0]}.{cc[1]}")
print(f"Available FA impls: {list_flash_attention_impls()}")
print(f"Warmup: {WARMUP}  Iters: {ITERS}  dtype: {DTYPE}")
print()


def benchmark(fn, warmup=WARMUP, iters=ITERS):
    """Return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# --------------------------------------------------------------------------
# dispatch_attention_fn benchmarks
# --------------------------------------------------------------------------
print("=" * 80)
print("dispatch_attention_fn benchmarks")
print("=" * 80)

from diffusers.models.attention_dispatch import dispatch_attention_fn

dispatch_backends = []

# Auto (None) — no FA activation
dispatch_backends.append(("None (no FA)", None, None))

# Auto (None) — with FA4 activated (aten override)
dispatch_backends.append(("None + FA4 activated", None, "FA4"))

# _native_flash — no FA activation
dispatch_backends.append(("_native_flash (no FA)", "_native_flash", None))

# _native_flash — with FA4 activated
dispatch_backends.append(("_native_flash + FA4", "_native_flash", "FA4"))

# _native_cudnn
dispatch_backends.append(("_native_cudnn", "_native_cudnn", None))

# _flash_3 (FA3 Hopper) — if available
try:
    from diffusers.utils.import_utils import is_flash_attn_3_available
    if is_flash_attn_3_available():
        dispatch_backends.append(("_flash_3 (FA3 Hopper)", "_flash_3", None))
    else:
        print("  [skip] _flash_3: flash_attn_3 not available")
except Exception as e:
    print(f"  [skip] _flash_3: {e}")

# flash (FA2) — if available
try:
    from diffusers.utils.import_utils import is_flash_attn_available
    if is_flash_attn_available():
        dispatch_backends.append(("flash (FA2)", "flash", None))
    else:
        print("  [skip] flash: flash_attn not available")
except Exception:
    pass

for B, H, S, D in SHAPES:
    print(f"\n--- shape: B={B} H={H} S={S} D={D} ---")
    print(f"{'Backend':<35} {'Forward (ms)':>14} {'Fwd+Bwd (ms)':>14}")
    print("-" * 65)

    for label, backend, fa_impl in dispatch_backends:
        # Set FA impl
        try:
            restore_flash_attention_impl(_raise_warn=False)
        except Exception:
            pass
        if fa_impl:
            try:
                activate_flash_attention_impl(fa_impl)
            except Exception as e:
                print(f"{label:<35} {'SKIP: ' + str(e)[:30]:>14}")
                continue

        q = torch.randn(B, S, H, D, dtype=DTYPE, device="cuda")
        k = torch.randn(B, S, H, D, dtype=DTYPE, device="cuda")
        v = torch.randn(B, S, H, D, dtype=DTYPE, device="cuda")

        # Forward only
        try:
            fwd_ms = benchmark(lambda: dispatch_attention_fn(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, backend=backend,
            ))
        except Exception as e:
            print(f"{label:<35} {'ERR: ' + str(e)[:40]:>14}")
            continue

        # Forward + Backward
        try:
            def fwd_bwd():
                q_ = q.detach().requires_grad_(True)
                k_ = k.detach().requires_grad_(True)
                v_ = v.detach().requires_grad_(True)
                out = dispatch_attention_fn(
                    q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=False, backend=backend,
                )
                out.sum().backward()
            fwdbwd_ms = benchmark(fwd_bwd)
        except Exception as e:
            fwdbwd_ms = f"ERR: {str(e)[:30]}"

        fwd_str = f"{fwd_ms:.2f}" if isinstance(fwd_ms, float) else fwd_ms
        bwd_str = f"{fwdbwd_ms:.2f}" if isinstance(fwdbwd_ms, float) else fwdbwd_ms
        print(f"{label:<35} {fwd_str:>14} {bwd_str:>14}")


# --------------------------------------------------------------------------
# flex_attention benchmarks
# --------------------------------------------------------------------------
print()
print("=" * 80)
print("flex_attention benchmarks (torch.compile)")
print("=" * 80)

flex_configs = [
    ("Triton (no FA)", None),
    ("FA4", "FA4"),
]

for B, H, S, D in SHAPES:
    print(f"\n--- shape: B={B} H={H} S={S} D={D} ---")
    # flex_attention expects (B, H, S, D) layout
    print(f"{'Backend':<35} {'Forward (ms)':>14} {'Fwd+Bwd (ms)':>14}")
    print("-" * 65)

    for label, fa_impl in flex_configs:
        try:
            restore_flash_attention_impl(_raise_warn=False)
        except Exception:
            pass
        if fa_impl:
            try:
                activate_flash_attention_impl(fa_impl)
            except Exception as e:
                print(f"{label:<35} {'SKIP: ' + str(e)[:30]:>14}")
                continue

        q = torch.randn(B, H, S, D, dtype=DTYPE, device="cuda")
        k = torch.randn(B, H, S, D, dtype=DTYPE, device="cuda")
        v = torch.randn(B, H, S, D, dtype=DTYPE, device="cuda")

        # Forward only
        try:
            fwd_ms = benchmark(lambda: flex_attention_compiled(q, k, v))
        except Exception as e:
            print(f"{label:<35} {'ERR: ' + str(e)[:40]:>14}")
            continue

        # Forward + Backward
        try:
            def fwd_bwd_flex():
                q_ = q.detach().requires_grad_(True)
                k_ = k.detach().requires_grad_(True)
                v_ = v.detach().requires_grad_(True)
                out = flex_attention_compiled(q_, k_, v_)
                out.sum().backward()
            fwdbwd_ms = benchmark(fwd_bwd_flex)
        except Exception as e:
            fwdbwd_ms = f"ERR: {str(e)[:30]}"

        fwd_str = f"{fwd_ms:.2f}" if isinstance(fwd_ms, float) else fwd_ms
        bwd_str = f"{fwdbwd_ms:.2f}" if isinstance(fwdbwd_ms, float) else fwdbwd_ms
        print(f"{label:<35} {fwd_str:>14} {bwd_str:>14}")

print()
print("=" * 80)
print("Done.")
