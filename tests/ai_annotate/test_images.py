"""Model images: checked from the header before any decoding."""

import io
import struct
import zlib

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
