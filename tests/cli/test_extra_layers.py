"""A YAML's extra_layers are shown in the viewer beside the raw data."""

import numpy as np
import pytest
import yaml
import zarr
from click.testing import CliRunner

from cellmap_flow.cli import yaml_cli
from cellmap_flow.globals import g


def _array(path, dtype):
    arr = zarr.open_group(str(path.parent), mode="a").create_dataset(
        path.name, data=np.zeros((4, 4, 4), dtype)
    )
    arr.attrs["resolution"] = [8, 8, 8]
    arr.attrs["offset"] = [0, 0, 0]
    return str(path)


def _config(tmp_path, extra_layers):
    path = tmp_path / "c.yaml"
    raw = _array(tmp_path / "raw.zarr" / "raw", np.uint8)
    path.write_text(yaml.safe_dump(
        {"data_path": raw, "charge_group": "grp", "models": {}, "extra_layers": extra_layers}
    ))
    return str(path)


def test_the_yaml_extra_layers_are_added_to_the_viewer(tmp_path, monkeypatch):
    from cellmap_flow.utils import neuroglancer_utils

    layers = {}

    class FakeViewer:
        def txn(self):
            return self

        def __enter__(self):
            return type("State", (), {"layers": layers})()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(neuroglancer_utils.neuroglancer, "Viewer", FakeViewer)
    monkeypatch.setattr(neuroglancer_utils, "create_and_run_app", lambda **k: "url")
    monkeypatch.setattr(yaml_cli, "install_cleanup_handlers", lambda: None)
    config = _config(tmp_path, [
        {"name": "pred", "path": _array(tmp_path / "pred.zarr" / "mito", np.uint8),
         "shader": "void main() { emitGrayscale(1.0); }", "blend": "additive"},
        {"name": "ids", "path": _array(tmp_path / "ids.zarr" / "s0", np.uint64),
         "layer_type": "segmentation", "disable_meshes": True},
    ])

    result = CliRunner().invoke(yaml_cli.main, [config])

    assert result.exit_code == 0, result.output
    assert list(layers) == ["data", "pred", "ids"]
    pred, ids = layers["pred"].to_json(), layers["ids"].to_json()
    assert (pred["type"], pred["blend"], pred["shader"]) == (
        "image", "additive", "void main() { emitGrayscale(1.0); }"
    )
    assert ids["type"] == "segmentation"
    assert ids["source"][0]["subsources"] == {"meshes": False}


@pytest.mark.parametrize("entry", [
    {"path": "/x.zarr"},
    {"name": "data", "path": "/x.zarr"},
    {"name": "x", "path": "/x.zarr", "layer_type": "points"},
])
def test_validate_only_refuses_a_bad_entry(tmp_path, entry):
    result = CliRunner().invoke(yaml_cli.main, [_config(tmp_path, [entry]), "--validate-only"])
    assert result.exit_code != 0 and "extra_layers" in result.output
    assert g.extra_layers == {}
