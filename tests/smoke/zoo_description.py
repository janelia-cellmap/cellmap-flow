"""Exits 0 when a BioImage Model Zoo description loads in this environment.

Run with the bioimageio environment's python (the macOS CI job does;
test_smoke_scripts does where bioimageio.spec is installed). Offline: a
minimal description and stand-in files, loaded as BioModelConfig loads a
zoo model's (format_version "latest", no IO checks). The Mac's bioimageio
environment once locked pydantic 2.14, which refuses an argument
bioimageio.spec 0.5.12.0 passes it, so every description failed to load:
"'index_key' is expected to be an integer or a string".
"""

import pathlib
import tempfile
import textwrap

import numpy as np
from bioimageio.spec import load_model_description

RDF = textwrap.dedent("""
    format_version: 0.5.4
    type: model
    name: smoke
    description: A tiny model description, loaded offline.
    authors: [{name: smoke}]
    cite: [{text: smoke, doi: "10.5281/zenodo.5764892"}]
    license: MIT
    documentation: README.md
    inputs:
      - id: raw
        axes:
          - {type: batch}
          - {type: channel, channel_names: [raw]}
          - {type: space, id: y, size: 16}
          - {type: space, id: x, size: 16}
        test_tensor: {source: test_input.npy}
    outputs:
      - id: out
        axes:
          - {type: batch}
          - {type: channel, channel_names: [out]}
          - {type: space, id: y, size: {tensor_id: raw, axis_id: y}}
          - {type: space, id: x, size: {tensor_id: raw, axis_id: x}}
        test_tensor: {source: test_output.npy}
    weights:
      torchscript: {source: weights.pt, pytorch_version: "2.0"}
""")

with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    (root / "rdf.yaml").write_text(RDF)
    (root / "README.md").write_text("smoke\n")
    for name in ("test_input.npy", "test_output.npy"):
        np.save(root / name, np.zeros((1, 1, 16, 16), np.float32))
    (root / "weights.pt").write_bytes(b"not loaded: the IO checks are off")
    description = load_model_description(root / "rdf.yaml", format_version="latest", perform_io_checks=False)
    print(f"loaded the {type(description).__name__} {description.name!r}: ok")
