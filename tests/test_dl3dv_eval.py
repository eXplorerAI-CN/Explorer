# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
import types

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.device = object
    sys.modules["torch"] = torch_stub
if "torchvision" not in sys.modules:
    torchvision_stub = types.ModuleType("torchvision")
    transforms_stub = types.ModuleType("torchvision.transforms")
    torchvision_stub.transforms = transforms_stub
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = transforms_stub
if "torchmetrics" not in sys.modules:
    torchmetrics_stub = types.ModuleType("torchmetrics")
    image_stub = types.ModuleType("torchmetrics.image")
    fid_stub = types.ModuleType("torchmetrics.image.fid")

    class FrechetInceptionDistance:  # pragma: no cover - only needed to import metrics_utils on login nodes.
        pass

    fid_stub.FrechetInceptionDistance = FrechetInceptionDistance
    image_stub.fid = fid_stub
    torchmetrics_stub.image = image_stub
    sys.modules["torchmetrics"] = torchmetrics_stub
    sys.modules["torchmetrics.image"] = image_stub
    sys.modules["torchmetrics.image.fid"] = fid_stub

from model_eval.metrics_utils import _scene_frame_root


def test_release_readme_uses_main_dl3dv_dataset_for_eval():
    readme = Path("README.md").read_text()

    assert "DL3DV/DL3DV-ALL-960P" in readme
    assert "DL3DV-ALL-960P" in readme
    assert "nerfstudio/transforms.json" not in readme


def test_release_readme_does_not_document_frame_source_override():
    readme = Path("README.md").read_text()

    assert "DL3DV_FRAME_SOURCE_DIR" not in readme
    assert "frame_source_dl3dv" not in readme
    assert "dl3dv_frame_source" not in readme


def test_dl3dv_frame_source_override_is_not_exposed():
    run_inference = Path("model_eval/run_inference.py").read_text()
    evalsets = Path("model_eval/dl3dv_reconstruction_evalsets.py").read_text()
    readme = Path("README.md").read_text()

    assert "dl3dv_frame_source" not in run_inference
    assert "dl3dv_frame_source" not in evalsets
    assert "DL3DV_FRAME_SOURCE" not in readme


def test_scene_frame_root_uses_main_dl3dv_scene_layout(tmp_path):
    scene_id = "scene-a"
    scene_dir = tmp_path / scene_id
    scene_dir.mkdir()
    (scene_dir / "transforms.json").write_text("{}")

    assert _scene_frame_root(tmp_path, scene_id) == scene_dir


def test_scene_frame_root_does_not_probe_benchmark_nerfstudio_layout(tmp_path):
    scene_id = "scene-a"
    nerfstudio_dir = tmp_path / scene_id / "nerfstudio"
    nerfstudio_dir.mkdir(parents=True)
    (nerfstudio_dir / "transforms.json").write_text("{}")

    assert _scene_frame_root(tmp_path, scene_id) is None
