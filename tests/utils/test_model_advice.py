"""What the dashboard advises for a model's input normalization (/api/model_advice)."""

from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.models.configs.cellpose import CellposeModelConfig
from cellmap_flow.models.configs.finetune import FinetuneModelConfig
from cellmap_flow.pipeline_spec import PipelineSpec

CELLPOSE = {"type": "cellpose", "voxel_size": [256, 256, 256], "name": "cellpose_sam_v2"}


@pytest.fixture
def verdict(dashboard, monkeypatch):
    """/api/model_advice's entry for one served model, with the chain's input
    normalization set to ``input_norm`` (step dicts, as the form sends them)
    first and no postprocessing."""
    from cellmap_flow.dashboard.routes import model_advice

    monkeypatch.setattr(model_advice, "_fetch_probe", lambda host: {"available": False})
    monkeypatch.setattr(model_advice, "raw_dtype_and_range", lambda path: ("uint8", (0, 255)))

    def verdict(model_config, input_norm=()):
        session = get_session()
        session.models_config = [model_config]
        session.jobs = [SimpleNamespace(model_name=model_config.name, host="http://gpu:8000")]
        session.set_pipeline(PipelineSpec(input_norm, ()))
        (entry,) = dashboard.get("/api/model_advice").get_json()["models"]
        return entry

    return verdict


@pytest.fixture
def advice(verdict):
    """The input normalization suggested for a model, with an empty chain."""
    return lambda model_config: verdict(model_config)["input_norm_suggestion"]


@pytest.fixture
def declaring_model(tmp_path):
    """An onnx model whose manifest declares MinMax 0..255 then 2*x-1."""
    import json

    from cellmap_flow.models.configs.onnx import MANIFEST_KEY
    from cellmap_flow.models.models_config import OnnxModelConfig
    from tests.utils.onnx_helpers import FINETUNED_MANIFEST, HEAD, with_metadata

    path = tmp_path / "ft.onnx"
    path.write_bytes(with_metadata(HEAD.read_bytes(), {MANIFEST_KEY: json.dumps(FINETUNED_MANIFEST)}))
    return OnnxModelConfig(path=str(path), name="ft")


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


def test_an_onnx_models_manifest_declares_its_input_normalization(advice, declaring_model):
    """The browser feeds an export through its manifest's input_norm; the
    server's advice offers the same steps, in order, as declared."""
    from tests.utils.onnx_helpers import HEAD
    from cellmap_flow.models.models_config import OnnxModelConfig

    suggestion = advice(declaring_model)
    assert suggestion["order"] == ["MinMaxNormalizer", "LambdaNormalizer"]
    assert suggestion["input_norm"] == {"MinMaxNormalizer": {"min_value": 0, "max_value": 255},
                                        "LambdaNormalizer": {"expression": "2*x-1"}}
    assert suggestion["confidence"] == "high" and "Declared" in suggestion["reason"]
    # Without a manifest it declares nothing, and the usual assumption stands.
    assert advice(OnnxModelConfig(path=str(HEAD), name="plain"))["confidence"] == "low"


def test_a_declared_value_edited_is_not_a_match(verdict, declaring_model):
    """The banner said the input normalization "looks right" for a manifest's
    steps whatever their values: the declared two steps with 2*x for the
    declared 2*x-1 feed the model something else. The values count, numbers
    by value as the form spells them."""
    as_form = [{"name": "MinMaxNormalizer", "min_value": "0.0", "max_value": "255.0", "invert": "False"},
               {"name": "LambdaNormalizer", "expression": "2*x-1"}]
    declared = verdict(declaring_model, as_form)
    assert declared["input_norm_matches"] is True and declared["input_norm_differences"] == []
    # The same map written another way is the declared one.
    rewritten = verdict(declaring_model, [as_form[0], {"name": "LambdaNormalizer", "expression": "x*2 - 1"}])
    assert rewritten["input_norm_matches"] is True and rewritten["input_norm_differences"] == []

    edited = verdict(declaring_model, [as_form[0], {"name": "LambdaNormalizer", "expression": "2*x"}])
    assert edited["input_norm_matches"] is False
    assert edited["input_norm_differences"] == [
        {"step": "LambdaNormalizer", "param": "expression", "configured": "2*x", "declared": "2*x-1"},
    ]
    # Other steps: the names say so, and there is no value to point at.
    assert verdict(declaring_model, [as_form[1]])["input_norm_differences"] == []
