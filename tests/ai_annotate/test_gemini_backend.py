"""Tests for the Gemini recolor backend and mask extraction, with a mocked
google.genai client so these run without network access or credentials.
"""

import io
import sys
import types

import numpy as np
import pytest
from PIL import Image

from cellmap_flow.ai_annotate.mask_extraction import extract_mask, slice_to_rgb
from cellmap_flow.ai_annotate.prompts import build_recolor_prompt


def _make_fake_genai_module(response_image: Image.Image | None, rate_limit_first_call=False):
    """Build a fake `google.genai` module tree and install it in sys.modules."""
    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    errors_mod = types.ModuleType("google.genai.errors")

    class ClientError(Exception):
        pass

    errors_mod.ClientError = ClientError

    class FakePart:
        def __init__(self, image):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            self.inline_data = types.SimpleNamespace(data=buf.getvalue())

    class FakeCandidate:
        def __init__(self, image):
            self.content = types.SimpleNamespace(parts=[FakePart(image)])

    class FakeResponse:
        def __init__(self, image):
            self.candidates = [FakeCandidate(image)]
            self.text = ""

    calls = {"count": 0}

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls["count"] += 1
            if rate_limit_first_call and calls["count"] == 1:
                raise ClientError("429 rate limited")
            return FakeResponse(response_image)

    class FakeClient:
        def __init__(self, *a, **kw):
            self.models = FakeModels()

    genai_mod.Client = FakeClient
    genai_mod.types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kw: kw,
    )
    genai_mod.errors = errors_mod

    google_mod.genai = genai_mod
    sys.modules["google"] = google_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.errors"] = errors_mod
    return calls


def test_build_recolor_prompt():
    prompt = build_recolor_prompt("mitochondria", "bright red")
    assert "mitochondria" in prompt
    assert "bright red" in prompt
    assert "segmentation mask" in prompt


def test_slice_to_rgb_normalizes_and_stacks():
    data = (np.random.rand(16, 16) * 1000).astype(np.uint16)
    img = slice_to_rgb(data)
    assert img.mode == "RGB"
    assert img.size == (16, 16)


def test_extract_mask_isolates_target_color():
    input_image = Image.new("RGB", (8, 8), (100, 100, 100))
    output = np.zeros((8, 8, 3), dtype=np.uint8)
    output[:4, :4] = (255, 0, 0)  # recolored region
    output_image = Image.fromarray(output, mode="RGB")

    mask = extract_mask(input_image, output_image, target_rgb=(255, 0, 0), threshold=200.0)

    assert mask.dtype == np.uint8
    assert (mask[:4, :4] == 255).all()
    assert (mask[4:, 4:] == 0).all()


def test_generate_recolored_image_returns_resized_output(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project")
    recolored = Image.new("RGB", (32, 32), (255, 0, 0))
    _make_fake_genai_module(recolored)

    from cellmap_flow.ai_annotate.gemini_backend import generate_recolored_image

    original = Image.new("RGB", (16, 16), (50, 50, 50))
    result = generate_recolored_image(original, "prompt text")

    assert result.size == original.size


def test_generate_recolored_image_retries_on_429(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project")
    recolored = Image.new("RGB", (16, 16), (255, 0, 0))

    import cellmap_flow.ai_annotate.gemini_backend as gb

    monkeypatch.setattr(gb.time, "sleep", lambda s: None)
    calls = _make_fake_genai_module(recolored, rate_limit_first_call=True)

    original = Image.new("RGB", (16, 16), (50, 50, 50))
    result = gb.generate_recolored_image(original, "prompt text")

    assert result.size == original.size
    assert calls["count"] == 2


def test_generate_recolored_image_requires_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    _make_fake_genai_module(Image.new("RGB", (8, 8)))

    from cellmap_flow.ai_annotate.gemini_backend import generate_recolored_image

    with pytest.raises(ValueError):
        generate_recolored_image(Image.new("RGB", (8, 8)), "prompt", vertex_project=None)
