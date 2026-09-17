# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from model_eval.export_checkpoint_pt import build_metadata, export_checkpoint, metadata_path_for_output


class ExportCheckpointPtTests(unittest.TestCase):
    def test_metadata_path_replaces_pt_suffix(self):
        self.assertEqual(
            metadata_path_for_output(Path("/exports/artifixer-s3-dmd-14b.pt")),
            Path("/exports/artifixer-s3-dmd-14b.metadata.json"),
        )

    def test_build_metadata_records_release_source(self):
        args = Namespace(
            checkpoint_dir=Path("/source/checkpoint_400/pytorch_model_fsdp_2"),
            output_pt=Path("/exports/artifixer-s3-dmd-14b-sr2he0ue-ckpt400-fsdp2.pt"),
            run_id="sr2he0ue",
            checkpoint=400,
            slot="pytorch_model_fsdp_2",
            source_path=Path("/canonical/checkpoint_400/pytorch_model_fsdp_2"),
            model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        )

        metadata = build_metadata(
            args,
            branch="ht-demo-tnt-eval-20260520",
            commit="fa6097322931e2a9c61a7295b664f214062fb9ab",
            export_date=datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(metadata["run_id"], "sr2he0ue")
        self.assertEqual(metadata["checkpoint"], 400)
        self.assertEqual(metadata["slot"], "pytorch_model_fsdp_2")
        self.assertEqual(metadata["source_path"], "/canonical/checkpoint_400/pytorch_model_fsdp_2")
        self.assertEqual(metadata["branch"], "ht-demo-tnt-eval-20260520")
        self.assertEqual(metadata["commit"], "fa6097322931e2a9c61a7295b664f214062fb9ab")
        self.assertEqual(metadata["export_date"], "2026-06-03T12:00:00+00:00")

    def test_export_checkpoint_saves_loaded_artifixer_state_dict(self):
        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

        class FakeTransformer:
            def state_dict(self):
                return {"blocks.0.opacity_embedding.weight": FakeTensor()}

        class FakeTorch:
            @staticmethod
            def is_tensor(value):
                return isinstance(value, FakeTensor)

            @staticmethod
            def save(state_dict, output_path):
                saved["path"] = output_path
                saved["keys"] = sorted(state_dict)

        with TemporaryDirectory() as tmpdir:
            args = Namespace(
                checkpoint_dir=Path("/source/checkpoint_400/pytorch_model_fsdp_2"),
                output_pt=Path(tmpdir) / "model.pt",
                metadata_path=None,
                run_id="sr2he0ue",
                checkpoint=400,
                slot="pytorch_model_fsdp_2",
                source_path=None,
                model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                branch="branch",
                commit="commit",
                overwrite=False,
            )
            saved = {}
            loaded = []
            transformer = FakeTransformer()

            export_checkpoint(
                args,
                transformer_factory=lambda parsed_args: transformer,
                dcp_loader=lambda model, checkpoint_dir: loaded.append((model, checkpoint_dir)),
                torch_module=FakeTorch,
            )

        self.assertEqual(loaded, [(transformer, args.checkpoint_dir)])
        self.assertEqual(saved["path"], args.output_pt)
        self.assertEqual(saved["keys"], ["blocks.0.opacity_embedding.weight"])


if __name__ == "__main__":
    unittest.main()
