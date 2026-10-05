"""Gemini image models on Google Cloud Vertex AI, asked to paint the structure.

The plane is sent as a PNG at its own size (padded to a square when it is
not one, see ``_square``) with the recolor prompt; the
model answers with an image in which the structure is painted the prompt's
colour on black, and the mask is the pixels of that colour. The plane is
not upsampled first: gemini-3-pro-image charges the same ~560 tokens for an
image whatever its size, so more pixels buy nothing, and the resolution the
prompt states must be the one sent.

Authentication is Application Default Credentials (``gcloud auth
application-default login``, or a service account on a cloud host), found
by the SDK, or the credentials file the config names (``credentials_file``,
read by ``secrets.load_google_credentials``), so this backend holds no key. The google-genai SDK is
imported only when the first call is made, so the dashboard runs without it
when the feature is off.

Every call has a timeout. Only "try again later" answers (429, 503) are
retried, at most ``MAX_ATTEMPTS`` times in all with bounded waits; anything
else fails at once. Failures become ``AIAnnotateError`` with a message
written here: the SDK's text goes to the server log, not the browser.
"""

import io
import logging
import re
import threading
import time

import numpy as np
from PIL import Image

from cellmap_flow.ai_annotate.backends.base import SegmentRequest, SegmentResult
from cellmap_flow.ai_annotate.errors import INSTALL_HINT, AIAnnotateError
from cellmap_flow.ai_annotate.images import decode_model_image
from cellmap_flow.ai_annotate.mask_extraction import extract_mask

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
# Waits before the second and third attempts. Vertex's 429 is usually a
# per-minute quota, so a few seconds rarely helps; these are long enough to
# matter and short enough that the user is not left waiting minutes.
RETRY_WAITS_S = (10.0, 30.0)
RETRY_CODES = (429, 503)
RETRY_STATUSES = ("UNAVAILABLE", "RESOURCE_EXHAUSTED")

# How much of the model's text reply a "no image" error may quote.
MAX_REPLY_CHARS = 200

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class VertexGeminiBackend:
    """Recolor-and-threshold segmentation through Vertex AI.

    ``project`` may be None, in which case the SDK takes it from
    ``GOOGLE_CLOUD_PROJECT`` or the credentials. ``credentials_file`` may be
    None, in which case the SDK finds Application Default Credentials
    itself. ``sleep`` is how retries
    wait and ``client_factory`` (no arguments, returns an object with
    ``.models.generate_content``) replaces the SDK client; both are for tests.
    """

    def __init__(self, project, location, timeout_s, sleep=time.sleep, client_factory=None, credentials_file=None):
        self.project = project
        self.credentials_file = credentials_file
        self.location = location
        self.timeout_s = timeout_s
        self._sleep = sleep
        self._client_factory = client_factory or self._make_client
        self._client = None
        self._client_lock = threading.Lock()

    def _make_client(self):
        try:
            from google import genai
        except ImportError:
            raise AIAnnotateError("unavailable", INSTALL_HINT) from None
        credentials = None
        if self.credentials_file:
            from cellmap_flow.ai_annotate.secrets import load_google_credentials

            credentials = load_google_credentials(self.credentials_file)
        # Retries are left to the SDK's default (none) and done here, where
        # only 429/503 are retried and the waits are bounded.
        return genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
            credentials=credentials,
            http_options=genai.types.HttpOptions(timeout=int(self.timeout_s * 1000)),
        )

    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                try:
                    self._client = self._client_factory()
                except AIAnnotateError:
                    raise
                except Exception as e:
                    raise self._user_error(e, model=None) from None
            return self._client

    def segment(self, request: SegmentRequest, model: str) -> SegmentResult:
        client = self._get_client()
        sent = _square(request.image.convert("RGB"))
        buffer = io.BytesIO()
        sent.save(buffer, format="PNG")
        # Plain dicts, which the SDK validates into its own types, so a test
        # client needs no SDK installed. The image goes first, then the text.
        contents = [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": buffer.getvalue()}},
                    {"text": request.prompt},
                ],
            }
        ]
        config = {"response_modalities": ["IMAGE", "TEXT"]}
        response = self._generate(client, model, contents, config)

        image_bytes, reply_text = _image_and_text(response)
        if image_bytes is None:
            raise AIAnnotateError("bad_response", _no_image_message(response, reply_text))
        model_image = decode_model_image(image_bytes)
        if model_image.size != sent.size:
            _warn_if_aspect_differs(model_image.size, sent.size)
            model_image = model_image.resize(sent.size, Image.BILINEAR)
        # Drop the padding _square added: what is left lines up with the plane.
        model_image = model_image.crop((0, 0, *request.image.size))
        mask = extract_mask(model_image, request.target_rgb)
        return SegmentResult(mask=mask, model_image=model_image, model=model, usage=_usage(response))

    def _generate(self, client, model, contents, config):
        """``generate_content``, retried on 429/503 only, at most ``MAX_ATTEMPTS`` times."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return client.models.generate_content(model=model, contents=contents, config=config)
            except Exception as e:
                if attempt < MAX_ATTEMPTS and _is_retryable(e):
                    wait = RETRY_WAITS_S[min(attempt - 1, len(RETRY_WAITS_S) - 1)]
                    logger.warning(
                        "Vertex AI answered %s (attempt %d of %d); retrying in %.0f s",
                        _code(e),
                        attempt,
                        MAX_ATTEMPTS,
                        wait,
                    )
                    self._sleep(wait)
                    continue
                raise self._user_error(e, model=model, attempts=attempt) from None

    def _user_error(self, exc, *, model, attempts=1):
        """``exc`` as an ``AIAnnotateError`` with a message written here.

        The SDK's own text (status, details, response body) is logged for
        whoever runs the dashboard, through the redacting filter, and never
        copied into the message the user sees.
        """
        logger.error("Vertex AI call failed: %s: %s", type(exc).__name__, exc)
        code = _code(exc)
        names = {cls.__name__ for cls in type(exc).__mro__}
        if code is None and any("Timeout" in name for name in names):
            return AIAnnotateError("timeout", f"Vertex AI did not answer within {self.timeout_s:g} s.")
        if code is None and any(cls.__module__.startswith("google.auth") for cls in type(exc).__mro__):
            # google.auth's errors: no credentials found, or a token that
            # could not be refreshed. Its TransportError is the network.
            if "TransportError" in names:
                return AIAnnotateError("unavailable", "Could not reach Google Cloud to get credentials.")
            if "RefreshError" in names:
                # Found, but Google would not renew them: a user login past
                # the organisation's re-sign-in period, or one revoked.
                where = (" and copy ~/.config/gcloud/application_default_credentials.json over the "
                         "config's credentials_file" if self.credentials_file else "")
                return AIAnnotateError(
                    "auth",
                    "Google asks you to sign in again: the saved login has expired. Run "
                    f"`gcloud auth application-default login`{where}, then try again.",
                )
            return AIAnnotateError(
                "auth",
                "No usable Google Cloud credentials on the dashboard's host: run "
                "`gcloud auth application-default login` there, then try again.",
            )
        if code in (401, 403):
            return AIAnnotateError(
                "auth",
                "Vertex AI refused the request's credentials: check that Application Default Credentials "
                "are set on the dashboard's host, that the project has the Vertex AI API enabled, and that "
                "you have the Vertex AI User role in it.",
            )
        if code == 404:
            return AIAnnotateError(
                "config",
                f"Vertex AI has no model {model} in location {self.location} for this project.",
            )
        if code == 429:
            return AIAnnotateError(
                "quota",
                f"Vertex AI's quota is used up (still refused after {attempts} attempts); wait a minute and try again.",
            )
        if code is not None and 500 <= code < 600:
            return AIAnnotateError("unavailable", f"Vertex AI is unavailable (HTTP {code}); try again later.")
        if code is not None and 400 <= code < 500:
            return AIAnnotateError("bad_response", f"Vertex AI rejected the request (HTTP {code}).")
        if names & {"ConnectError", "ConnectionError", "NetworkError"}:
            return AIAnnotateError("unavailable", "Could not reach Vertex AI from the dashboard's host.")
        if isinstance(exc, ValueError) and model is None:
            # genai.Client raises ValueError when it finds no project.
            return AIAnnotateError(
                "config",
                "No Google Cloud project for Vertex AI: set the provider's project in the AI-annotate "
                "config, or GOOGLE_CLOUD_PROJECT.",
            )
        return AIAnnotateError(
            "bad_response", f"The Vertex AI call failed ({type(exc).__name__}); see the dashboard's log."
        )


def _code(exc):
    """The HTTP status an SDK error carries (``google.genai.errors.APIError.code``), or None."""
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _is_retryable(exc):
    return _code(exc) in RETRY_CODES or getattr(exc, "status", None) in RETRY_STATUSES


def _parts(response):
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    return list(getattr(content, "parts", None) or [])


def _image_and_text(response):
    """The last image the model gave as its answer, and its text, from the first candidate.

    Parts marked ``thought`` are skipped: an image model that thinks first
    may send draft images before its answer.
    """
    image = None
    texts = []
    for part in _parts(response):
        if getattr(part, "thought", False):
            continue
        blob = getattr(part, "inline_data", None)
        if blob is not None and getattr(blob, "data", None):
            image = blob.data
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return image, " ".join(texts)


def _plain(text, limit=MAX_REPLY_CHARS):
    text = _CONTROL_CHARS.sub(" ", str(text)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _no_image_message(response, reply_text):
    """Why the reply had no image, from the block or finish reason and the model's own words."""
    message = "The model answered without an image"
    feedback = getattr(response, "prompt_feedback", None)
    block = getattr(feedback, "block_reason", None)
    candidates = getattr(response, "candidates", None) or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    if block:
        message += f" (blocked: {_plain(getattr(block, 'name', block), 40)})"
    elif finish and getattr(finish, "name", str(finish)) not in ("STOP", "FinishReason.STOP"):
        message += f" (stopped: {_plain(getattr(finish, 'name', finish), 40)})"
    if reply_text:
        message += f'. It said: "{_plain(reply_text)}"'
    return message + "."


def _usage(response):
    """Token counts the response reports, under provider-neutral names."""
    meta = getattr(response, "usage_metadata", None)
    usage = {}
    for ours, theirs in (
        ("prompt_tokens", "prompt_token_count"),
        ("output_tokens", "candidates_token_count"),
        ("total_tokens", "total_token_count"),
    ):
        value = getattr(meta, theirs, None)
        if isinstance(value, int):
            usage[ours] = value
    return usage


def _square(image):
    """``image`` padded on the right or bottom to a square, with its median grey.

    Gemini's image models answer in a few fixed aspect ratios, and a square
    one for a square input; a plane clipped at the volume's edge is not
    square, and resizing the answer back to it would stretch the mask. The
    padding is a flat grey rather than black, which the prompt asks the model
    to paint everything else, so it reads as nothing rather than as a
    structure's dark edge; ``segment`` crops it off the answer.
    """
    width, height = image.size
    side = max(width, height)
    if width == height:
        return image
    grey = int(np.median(np.asarray(image.convert("L"))))
    padded = Image.new("RGB", (side, side), (grey, grey, grey))
    padded.paste(image, (0, 0))
    return padded


def _warn_if_aspect_differs(got, sent):
    """Log when the model's image is not the sent image's shape, only bigger
    or smaller: resizing it back stretches the mask, which the preview shows."""
    if abs(got[0] / got[1] - sent[0] / sent[1]) > 0.02 * (sent[0] / sent[1]):
        logger.warning("The model returned a %dx%d image for a %dx%d plane; stretching it to fit", *got, *sent)
