"""ONNX models for the onnx type's tests: the browser tests' head.onnx, and a
manifest written into it as the browser's *Download finetuned model* writes one."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The browser finetune's fixture (web/test/fixtures/make_onnx_fixture.py):
# input float32 [1, 1, 1, 8, 8], output [1, 2, 1, 6, 6], a 2D student's tail.
HEAD = ROOT / "web" / "test" / "fixtures" / "head.onnx"


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return bytes(out)


def _length_delimited(number: int, payload: bytes) -> bytes:
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def with_metadata(model: bytes, entries: dict) -> bytes:
    """``model`` with ``entries`` appended to its metadata_props, as the browser's
    withMetadata (web/src/finetune/onnx-proto.ts) writes them: one field 14 each,
    a StringStringEntryProto of field 1 the key and field 2 the value, after
    every other field."""
    added = b"".join(
        _length_delimited(14, _length_delimited(1, key.encode()) + _length_delimited(2, value.encode()))
        for key, value in entries.items()
    )
    return model + added


# What the browser keeps for a model finetuned from setup_53_2d's head
# (finetunedEntry: the base's manifest under the new name, one variant whose
# file is <name>.onnx), shaped like the fixture.
FINETUNED_MANIFEST = {
    "model": "mito_aff_setup53-2d_16nm-ft1",
    "task": "mito_affinities",
    "architecture": "Unet3dIso (2D-collapsed)",
    "checkpoint": "finetuned in the browser from student_60000",
    "params": 1927907,
    "channels": ["mito_aff_z", "mito_aff_y"],
    "input_voxel_size_nm": [16, 16, 16],
    "output_voxel_size_nm": [16, 16, 16],
    "output": "raw logits (apply sigmoid / threshold at 0)",
    "input_norm": [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255},
                   {"name": "LambdaNormalizer", "expression": "2*x-1"}],
    "io": {"input": "input", "output": "output", "description": "input 'input' float32 [1,1,1,y,x]"},
    "variants": [{"variant": "fp32", "input_size": [1, 8, 8], "output_size": [1, 6, 6],
                  "file": "mito_aff_setup53-2d_16nm-ft1.onnx", "file_mb": 0.01}],
}
