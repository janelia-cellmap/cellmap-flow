"""Tests for finetuning dashboard service helpers."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cellmap_flow.dashboard.routes.finetune.service import (
    _autodetect_output_type,
    _build_restart_params,
)


class FinetuneServiceHelperTests(unittest.TestCase):
    def test_build_restart_params_maps_distillation_scope(self):
        params = _build_restart_params(
            {
                "batch_size": 4,
                "loss_type": "margin",
                "distillation_scope": "all",
                "offsets": [[1, 0, 0]],
            }
        )

        self.assertEqual(params["batch_size"], 4)
        self.assertEqual(params["loss_type"], "margin")
        self.assertEqual(params["distillation_all_voxels"], True)
        self.assertEqual(params["offsets"], [[1, 0, 0]])

    def test_autodetect_output_type_reads_script_offsets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "model.py"
            script_path.write_text("offsets = [[1, 0, 0], [0, 1, 0]]\n")
            model_config = SimpleNamespace(script_path=str(script_path))

            output_type, offsets = _autodetect_output_type(
                model_config,
                output_type=None,
                offsets=None,
            )

            self.assertEqual(output_type, "affinities")
            self.assertEqual(offsets, "[[1, 0, 0], [0, 1, 0]]")


if __name__ == "__main__":
    unittest.main()
