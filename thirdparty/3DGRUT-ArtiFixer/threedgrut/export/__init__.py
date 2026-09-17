# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from threedgrut.export.base import ExportableModel, ModelExporter
from threedgrut.export.ingp_exporter import INGPExporter
from threedgrut.export.ply_exporter import PLYExporter

__all__ = [
    "ExportableModel",
    "ModelExporter",
    "PLYExporter",
    "INGPExporter",
]

try:
    from threedgrut.export.usdz_exporter import USDZExporter

    __all__ += ["USDZExporter"]
except ModuleNotFoundError as e:
    if e.name is None or not e.name.startswith("pxr"):
        raise
    import warnings

    USDZExporter = None
    warnings.warn(f"USD exporter unavailable: {e}", ImportWarning, stacklevel=2)
