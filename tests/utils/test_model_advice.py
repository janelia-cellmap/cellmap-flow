"""What the dashboard advises for a model's input normalization (/api/model_advice)."""

from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.models.configs.cellpose import CellposeModelConfig
from cellmap_flow.models.configs.finetune import FinetuneModelConfig
from cellmap_flow.process_chain import process_chain

CELLPOSE = {"type": "cellpose", "voxel_size": [256, 256, 256], "name": "cellpose_sam_v2"}


@pytest.fixture
def advice(dashboard, monkeypatch):
    from cellmap_flow.dashboard.routes import model_advice

    monkeypatch.setattr(model_advice, "_fetch_probe", lambda host: {"available": False})
    monkeypatch.setattr(model_advice, "_raw_dtype_and_range", lambda path: ("uint8", (0, 255)))

    def advice(model_config):
        session = get_session()
        session.models_config = [model_config]
        session.jobs = [SimpleNamespace(model_name=model_config.name, host="http://gpu:8000")]
        process_chain().input_norms, process_chain().postprocess = [], []
        (entry,) = dashboard.get("/api/model_advice").get_json()["models"]
        return entry["input_norm_suggestion"]

    return advice


@pytest.mark.parametrize("model_config", [
    pytest.param(CellposeModelConfig(voxel_size=256, name="cellpose_sam_v2"), id="cellpose"),
    # The finetune in the report: Cellpose finetuned, served from its YAML.
    pytest.param(FinetuneModelConfig(lora_adapter_path="/runs/lora_adapter", base_model=CELLPOSE,
                                     name="cellpose_sam_v2_finetuned_20261014_133414"), id="a-finetune-of-cellpose"),
])
def test_a_model_that_normalizes_its_own_input_gets_no_suggestion(advice, model_config):
    """Cellpose normalizes each slice itself, and its finetune trained on raw
    values, but the page suggested adding [-1, 1] normalization for it."""
    assert advice(model_config)["input_norm"] is None


def test_a_model_that_declares_nothing_is_still_assumed_to_take_minus_one_to_one(advice):
    # A stand-in rather than a ModelConfig subclass: the registry lists every
    # subclass, and the type listings' tests would see one more.
    suggestion = advice(SimpleNamespace(name="mine", to_dict=dict))
    assert suggestion["order"] == ["MinMaxNormalizer", "LambdaNormalizer"] and suggestion["confidence"] == "low"


def test_an_onnx_models_manifest_declares_its_input_normalization(advice, tmp_path):
    """The browser feeds an export through its manifest's input_norm; the
    server's advice offers the same steps, in order, as declared."""
    import json

    from cellmap_flow.models.configs.onnx import MANIFEST_KEY
    from cellmap_flow.models.models_config import OnnxModelConfig
    from tests.utils.onnx_helpers import FINETUNED_MANIFEST, HEAD, with_metadata

    path = tmp_path / "ft.onnx"
    path.write_bytes(with_metadata(HEAD.read_bytes(), {MANIFEST_KEY: json.dumps(FINETUNED_MANIFEST)}))
    suggestion = advice(OnnxModelConfig(path=str(path), name="ft"))
    assert suggestion["order"] == ["MinMaxNormalizer", "LambdaNormalizer"]
    assert suggestion["input_norm"] == {"MinMaxNormalizer": {"min_value": 0, "max_value": 255},
                                        "LambdaNormalizer": {"expression": "2*x-1"}}
    assert suggestion["confidence"] == "high" and "Declared" in suggestion["reason"]
    # Without a manifest it declares nothing, and the usual assumption stands.
    assert advice(OnnxModelConfig(path=str(HEAD), name="plain"))["confidence"] == "low"
