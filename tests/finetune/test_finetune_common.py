"""Tests for the finetune dashboard helpers in routes/finetune/common.py."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.routes.finetune.common import (
    autodetect_output_type,
    build_restart_params,
)


class FinetuneCommonHelperTests(unittest.TestCase):
    def test_build_restart_params_maps_distillation_scope(self):
        params = build_restart_params(
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
        # JSON, as --offsets is: a list made the trainer's json.loads() fail.
        self.assertEqual(params["offsets"], "[[1, 0, 0]]")

    def test_autodetect_output_type_reads_script_offsets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "model.py"
            script_path.write_text("offsets = [[1, 0, 0], [0, 1, 0]]\n")
            model_config = SimpleNamespace(script_path=str(script_path))

            output_type, offsets = autodetect_output_type(
                model_config,
                output_type=None,
                offsets=None,
            )

            self.assertEqual(output_type, "affinities")
            self.assertEqual(offsets, "[[1, 0, 0], [0, 1, 0]]")


MINIO = "http://10.0.0.5:9000/annotations/vol-1.zarr"
PROXY = "https://gateway.example.org/minio/annotations/vol-1.zarr"


@pytest.mark.parametrize(
    "template, headers, expected",
    [
        (None, {"X-Forwarded-Host": "gateway.example.org"}, MINIO),  # opt-in only
        ("{proto}://{host}/minio", {}, MINIO),  # direct access: no proxy, no rewrite
        ("{proto}://{host}/minio", None, MINIO),  # outside a request
        ("{proto}://{host}/minio",
         {"X-Forwarded-Host": "gateway.example.org, inner", "X-Forwarded-Proto": "https"}, PROXY),
        ("{proto}://{host}/minio", {"X-Forwarded-Host": "gateway.example.org"},
         PROXY.replace("https", "http")),  # no forwarded proto: the request's own
        ("https://gateway.example.org/minio/", {"X-Forwarded-Host": "anything"}, PROXY),
    ],
)
def test_minio_urls_are_rewritten_only_behind_a_configured_proxy(monkeypatch, template, headers, expected):
    from flask import Flask

    from cellmap_flow.dashboard.routes.finetune.common import rewrite_minio_url_for_proxy

    if template is None:
        monkeypatch.delenv("CELLMAP_FLOW_MINIO_PROXY_URL", raising=False)
    else:
        monkeypatch.setenv("CELLMAP_FLOW_MINIO_PROXY_URL", template)
    if headers is None:
        assert rewrite_minio_url_for_proxy(MINIO) == expected
        return
    with Flask(__name__).test_request_context(base_url="http://node:5000", headers=headers):
        assert rewrite_minio_url_for_proxy(MINIO) == expected


if __name__ == "__main__":
    unittest.main()
