"""Model images: checked from the header before any decoding, re-encoded for the browser."""

import base64
import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from cellmap_flow.ai_annotate import images
from cellmap_flow.ai_annotate.errors import AIAnnotateError


def _encode(image, fmt):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def _png_header_only(width, height):
    """A PNG whose header claims ``width`` x ``height`` and holds no pixel data."""

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_allowed_formats_decode_to_rgb(fmt):
    source = Image.new("RGBA" if fmt != "JPEG" else "L", (12, 7), 128)
    decoded = images.decode_model_image(_encode(source, fmt))
    assert decoded.mode == "RGB" and decoded.size == (12, 7)


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_other_formats_are_rejected(fmt):
    with pytest.raises(AIAnnotateError) as caught:
        images.decode_model_image(_encode(Image.new("RGB", (8, 8)), fmt))
    assert caught.value.category == "bad_response"


def test_an_oversized_image_is_rejected_before_decoding(monkeypatch):
    def no_load(self):
        raise AssertionError("pixels were decoded")

    monkeypatch.setattr(Image.Image, "load", no_load)
    with pytest.raises(AIAnnotateError) as caught:
        images.decode_model_image(_png_header_only(4097, 4096))
    assert caught.value.category == "bad_response"
    assert "4097x4096" in caught.value.user_message


def test_the_size_limit_is_inclusive():
    # A header at exactly the limit passes the size check and then fails to
    # decode (it has no pixels): the error is about decoding, not size.
    with pytest.raises(AIAnnotateError, match="could not be decoded"):
        images.decode_model_image(_png_header_only(4096, 4096))


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\ntruncated"])
def test_garbage_is_rejected(data):
    with pytest.raises(AIAnnotateError) as caught:
        images.decode_model_image(data)
    assert caught.value.category == "bad_response"


def test_png_base64_round_trips():
    pixels = np.random.default_rng(0).integers(0, 255, (5, 6, 3), dtype=np.uint8)
    text = images.png_base64(Image.fromarray(pixels))
    assert not text.startswith("data:")
    decoded = Image.open(io.BytesIO(base64.b64decode(text)))
    assert decoded.format == "PNG"
    np.testing.assert_array_equal(np.asarray(decoded), pixels)
