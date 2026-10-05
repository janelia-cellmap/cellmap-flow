"""Checking the images a model sends back, and encoding images for the browser.

A model's response is untrusted input: its image could be huge, malformed or
in a format whose decoder has a history of bugs. ``decode_model_image``
reads only the header first (PIL's ``open`` is lazy), checks the format and
size from it, and decodes the pixels only when both pass. Nothing the model
returned goes to the browser as is: ``png_base64`` re-encodes the decoded
pixels as a fresh PNG.
"""

import base64
import io
import warnings

from PIL import Image

from cellmap_flow.ai_annotate.errors import AIAnnotateError

MAX_PIXELS = 4096 * 4096
ALLOWED_FORMATS = ("PNG", "JPEG", "WEBP")


def decode_model_image(data):
    """The model's image as RGB, after checking its format and size before decoding.

    Raises ``AIAnnotateError("bad_response")`` for anything other than a PNG,
    JPEG or WebP of at most ``MAX_PIXELS`` pixels, or one that fails to decode.
    """
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise AIAnnotateError("bad_response", "The model returned an empty image.")
    try:
        # ``formats`` limits which decoders PIL even tries on these bytes.
        image = Image.open(io.BytesIO(bytes(data)), formats=list(ALLOWED_FORMATS))
    except Exception:
        raise AIAnnotateError("bad_response", "The model returned an image that is not a PNG, JPEG or WebP.") from None
    if image.format not in ALLOWED_FORMATS:
        raise AIAnnotateError("bad_response", "The model returned an image that is not a PNG, JPEG or WebP.")
    width, height = image.size
    if width < 1 or height < 1 or width * height > MAX_PIXELS:
        raise AIAnnotateError(
            "bad_response", f"The model returned a {width}x{height} image, larger than this dashboard accepts."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image.load()
        return image.convert("RGB")
    except Exception:
        raise AIAnnotateError("bad_response", "The model returned an image that could not be decoded.") from None


def png_base64(image):
    """``image`` as base64 PNG text (no ``data:`` prefix), for a JSON response."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
