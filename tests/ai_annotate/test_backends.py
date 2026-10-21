"""The Vertex Gemini provider against a stand-in client (no network), including
its retries and the errors users are shown."""

import io
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cellmap_flow.ai_annotate import secrets
from cellmap_flow.ai_annotate.backends.base import SegmentRequest
from cellmap_flow.ai_annotate.backends.vertex_gemini import MAX_ATTEMPTS, RETRY_WAITS_S, VertexGeminiBackend
from cellmap_flow.ai_annotate.errors import AIAnnotateError

RED = (255, 0, 0)
SDK_DETAIL = "INTERNAL-DETAIL projects/secret-proj/locations/global quota_metric=xyz"


def _plane(height=40, width=60):
    """A grey plane, bright with a dark square in rows 10-19, cols 20-34."""
    pixels = np.full((height, width), 200, dtype=np.uint8)
    pixels[10:20, 20:35] = 30
    return Image.fromarray(np.stack([pixels] * 3, axis=-1))


def _request(image=None, click=None):
    return SegmentRequest(image=image or _plane(), prompt="paint it", target_rgb=RED, click_px=click)


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _painted_square(side=60):
    """A good answer to the 60x40 plane, which is sent padded to 60x60: the
    dark square red at the plane's rows 10-20 and columns 20-35, scaled to
    ``side`` as a model answering bigger would."""
    pixels = np.zeros((side, side, 3), dtype=np.uint8)
    k = side // 60
    pixels[10 * k:20 * k, 20 * k:35 * k] = RED
    return Image.fromarray(pixels)


def _part(data=None, text=None, thought=None):
    blob = SimpleNamespace(data=data, mime_type="image/png") if data is not None else None
    return SimpleNamespace(inline_data=blob, text=text, thought=thought)


def _response(*parts, finish_reason="STOP"):
    candidate = SimpleNamespace(content=SimpleNamespace(parts=list(parts)), finish_reason=finish_reason)
    return SimpleNamespace(
        candidates=[candidate],
        prompt_feedback=SimpleNamespace(block_reason=None),
        usage_metadata=SimpleNamespace(prompt_token_count=600, candidates_token_count=1300, total_token_count=1900),
    )


class APIError(Exception):
    """Shaped like google.genai.errors.APIError: an HTTP ``code`` and a ``status``."""

    def __init__(self, code, status=None):
        super().__init__(f"{code} {status}. {SDK_DETAIL}")
        self.code = code
        self.status = status


class ReadTimeout(Exception):
    """Named like httpx's timeout, which the SDK lets through."""


class FakeClient:
    """Answers ``generate_content`` from a script: each item a response or an exception to raise."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _backend(client, sleeps=None):
    return VertexGeminiBackend(
        "proj", "global", 30, sleep=(sleeps.append if sleeps is not None else lambda s: None), client_factory=lambda: client
    )


def test_vertex_sends_the_plane_padded_square_and_reads_the_mask():
    client = FakeClient(_response(_part(text="Here you go."), _part(data=_png(_painted_square()))))
    result = _backend(client).segment(_request(), "gemini-3-pro-image")

    (call,) = client.calls
    assert call["model"] == "gemini-3-pro-image"
    assert call["config"] == {"response_modalities": ["IMAGE", "TEXT"]}
    parts = call["contents"][0]["parts"]
    assert parts[1] == {"text": "paint it"}
    sent = Image.open(io.BytesIO(parts[0]["inline_data"]["data"]))
    assert parts[0]["inline_data"]["mime_type"] == "image/png"
    # Padded to a square below the plane, with flat grey, and not upsampled.
    assert sent.size == (60, 60)
    padding = np.asarray(sent)[40:]
    assert (padding == padding[0, 0]).all() and 0 < padding[0, 0, 0] < 255
    np.testing.assert_array_equal(np.asarray(sent)[:40], np.asarray(_plane()))

    assert result.mask.dtype == bool and result.mask.shape == (40, 60)
    assert result.mask[10:20, 20:35].all() and result.mask.sum() == 150
    assert result.model_image.size == (60, 40)
    assert result.usage == {"prompt_tokens": 600, "output_tokens": 1300, "total_tokens": 1900}


def test_a_bigger_reply_is_resized_and_cropped_to_the_plane():
    client = FakeClient(_response(_part(data=_png(_painted_square(120)))))
    result = _backend(client).segment(_request(), "m")
    assert result.model_image.size == (60, 40) and result.mask.shape == (40, 60)
    assert result.mask[12:18, 22:33].all()


def test_draft_thought_images_are_skipped():
    draft = Image.new("RGB", (60, 40), RED)
    client = FakeClient(_response(_part(data=_png(draft), thought=True), _part(data=_png(_painted_square()))))
    assert _backend(client).segment(_request(), "m").mask.sum() == 150


def test_a_429_is_retried_with_bounded_waits_then_succeeds():
    sleeps = []
    client = FakeClient(APIError(429, "RESOURCE_EXHAUSTED"), APIError(503, "UNAVAILABLE"), _response(_part(data=_png(_painted_square()))))
    _backend(client, sleeps).segment(_request(), "m")
    assert len(client.calls) == 3
    assert sleeps == list(RETRY_WAITS_S[:2])
    assert all(0 < s <= 60 for s in sleeps)


def test_retries_stop_after_max_attempts_with_a_quota_error():
    sleeps = []
    client = FakeClient(*[APIError(429, "RESOURCE_EXHAUSTED") for _ in range(10)])
    with pytest.raises(AIAnnotateError) as caught:
        _backend(client, sleeps).segment(_request(), "m")
    assert len(client.calls) == MAX_ATTEMPTS == 3
    assert len(sleeps) == MAX_ATTEMPTS - 1
    assert (caught.value.category, caught.value.http_status) == ("quota", 429)


@pytest.mark.parametrize(
    "error, category",
    [
        (APIError(400, "INVALID_ARGUMENT"), "bad_response"),
        (APIError(401, "UNAUTHENTICATED"), "auth"),
        (APIError(403, "PERMISSION_DENIED"), "auth"),
        (APIError(404, "NOT_FOUND"), "config"),
        (APIError(500, "INTERNAL"), "unavailable"),
        (ReadTimeout("timed out " + SDK_DETAIL), "timeout"),
        (RuntimeError("weird " + SDK_DETAIL), "bad_response"),
    ],
)
def test_other_errors_are_not_retried_and_never_show_the_sdk_text(error, category, caplog):
    sleeps = []
    client = FakeClient(error, _response(_part(data=_png(_painted_square()))))
    with caplog.at_level(logging.ERROR, logger="cellmap_flow.ai_annotate"):
        with pytest.raises(AIAnnotateError) as caught:
            _backend(client, sleeps).segment(_request(), "m")
    assert len(client.calls) == 1 and sleeps == []
    assert caught.value.category == category
    assert "INTERNAL-DETAIL" not in caught.value.user_message
    assert "secret-proj" not in caught.value.user_message
    assert "INTERNAL-DETAIL" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert "INTERNAL-DETAIL" in caplog.text  # the detail goes to the server log


def test_a_secret_in_sdk_error_text_is_redacted_from_the_log():
    secrets._clear_registered_secrets()
    secrets.register_secret("sk-live-abcdefghijklmnop")
    logger = logging.getLogger("cellmap_flow")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    logger.addHandler(handler)
    secrets.install_log_redaction(logger)
    try:
        client = FakeClient(APIError(400, "key sk-live-abcdefghijklmnop rejected"))
        with pytest.raises(AIAnnotateError):
            _backend(client).segment(_request(), "m")
        assert "Vertex AI call failed" in buffer.getvalue()
        assert "sk-live-abcdefghijklmnop" not in buffer.getvalue()
    finally:
        logger.removeHandler(handler)
        secrets._clear_registered_secrets()


@pytest.mark.parametrize("credentials_file", [None, "/groups/lab/adc.json"])
def test_an_expired_login_says_to_sign_in_again(credentials_file):
    # What google.auth raises when Google wants the user to sign in again
    # ("Reauthentication is needed"), or the refresh token was revoked.
    error_type = type("RefreshError", (Exception,), {"__module__": "google.auth.exceptions"})
    backend = VertexGeminiBackend("p", "global", 30, credentials_file=credentials_file,
                                  client_factory=lambda: (_ for _ in ()).throw(error_type("Reauthentication is needed")))
    with pytest.raises(AIAnnotateError) as caught:
        backend.segment(_request(), "m")
    message = caught.value.user_message
    assert caught.value.category == "auth" and "gcloud auth application-default login" in message
    assert ("credentials_file" in message) == bool(credentials_file)


def test_no_image_is_a_bad_response_quoting_at_most_200_chars_of_plain_text():
    reply = "I can't do that. <script>alert(1)</script>\x00\x1b" + "x" * 500
    client = FakeClient(_response(_part(text=reply), finish_reason=SimpleNamespace(name="IMAGE_SAFETY")))
    with pytest.raises(AIAnnotateError) as caught:
        _backend(client).segment(_request(), "m")
    message = caught.value.user_message
    assert caught.value.category == "bad_response"
    assert "IMAGE_SAFETY" in message and "I can't do that." in message
    assert "\x00" not in message and "\x1b" not in message
    assert "x" * 201 not in message and len(message) < 320


def test_the_requests_we_build_are_valid_for_the_real_sdk(monkeypatch):
    """Where google-genai is installed: our dicts validate as its types, its
    errors classify as we expect, and the client is built with a timeout in ms."""
    genai = pytest.importorskip("google.genai")
    from google.genai import errors, types

    types.GenerateContentConfig(**{"response_modalities": ["IMAGE", "TEXT"]})
    client = FakeClient(_response(_part(data=_png(_painted_square()))))
    _backend(client).segment(_request(), "m")
    content = types.Content.model_validate(client.calls[0]["contents"][0])
    assert content.parts[0].inline_data.mime_type == "image/png"

    sleeps = []
    sdk_errors = [errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": SDK_DETAIL}})] * 3
    with pytest.raises(AIAnnotateError) as caught:
        _backend(FakeClient(*sdk_errors), sleeps).segment(_request(), "m")
    assert caught.value.category == "quota" and len(sleeps) == 2
    with pytest.raises(AIAnnotateError) as caught:
        _backend(FakeClient(errors.ClientError(403, {"error": {"message": SDK_DETAIL}}))).segment(_request(), "m")
    assert caught.value.category == "auth" and "INTERNAL-DETAIL" not in caught.value.user_message

    built = {}
    monkeypatch.setattr(genai, "Client", lambda **kwargs: built.update(kwargs) or FakeClient())
    VertexGeminiBackend("proj", "global", 45)._make_client()
    assert built["vertexai"] is True and built["project"] == "proj" and built["location"] == "global"
    assert built["http_options"].timeout == 45_000
    assert built["credentials"] is None  # the SDK finds Application Default Credentials itself


def test_a_configured_credentials_file_is_handed_to_the_client(monkeypatch, tmp_path):
    genai = pytest.importorskip("google.genai")
    path = tmp_path / "adc.json"
    path.write_text('{"type": "authorized_user", "client_id": "c", "client_secret": "s-secret-value", '
                    '"refresh_token": "r-refresh-token-value"}')
    path.chmod(0o600)
    built = {}
    monkeypatch.setattr(genai, "Client", lambda **kwargs: built.update(kwargs) or FakeClient())

    VertexGeminiBackend("proj", "global", 45, credentials_file=str(path))._make_client()

    assert built["credentials"].refresh_token == "r-refresh-token-value"
