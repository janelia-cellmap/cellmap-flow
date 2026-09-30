"""What passes between the dashboard and an inference server, as text.

A prediction layer's source is ``zarr://<server>/<model><ARGS_KEY><blob><ARGS_KEY>``.
The blob is the layer's chain, ``{INPUT_NORM_KEY: [...], POSTPROCESS_KEY:
[...]}`` followed by extras such as ``dashboard_url`` and ``digest``, as
minified JSON in URL-safe base64 without padding (``encode_to_str``). Every
request the viewer makes carries it, and ``split_dataset_url`` finds it in
the dataset part of a request's path. ``pipeline_spec.PipelineSpec`` reads
and writes the chain; this module is only the codec, so a server and a
dashboard of different versions still understand each other.

The other way, a server announces its address by printing it between the
two ``IP_PATTERN`` markers (``CellMapFlowServer.run``). They are defined in
``jobs.spec``, which finds them in a job's log.

The same codec carries a dict argument on a command line: a model entry
for the finetune CLI's ``--model-entry``, and a finetuned model's
``base_model`` for ``cellmap_flow_server``.
"""

import base64
import json
from typing import Optional

from cellmap_flow.jobs.spec import IP_PATTERN  # noqa: F401  (part of this protocol)

ARGS_KEY = "__CFLOW_ARGS__"
INPUT_NORM_KEY = "input_norm"
POSTPROCESS_KEY = "postprocess"


def encode_to_str(data):
    """Encodes a JSON object into a URL-safe string without '/', '+', or '='."""
    json_str = json.dumps(data, separators=(",", ":"))  # Minify JSON
    encoded_bytes = base64.urlsafe_b64encode(json_str.encode())  # Base64 encode
    return encoded_bytes.decode().rstrip("=")  # Remove padding ('=')


def decode_to_json(encoded_str):
    """Decodes a URL-safe string back into a JSON object."""
    padding_needed = 4 - (len(encoded_str) % 4)
    encoded_str += "=" * (padding_needed % 4)  # Add padding back if needed
    json_str = base64.urlsafe_b64decode(encoded_str.encode()).decode()  # Decode Base64
    return json.loads(json_str)  # Convert back to JSON


def split_dataset_url(dataset: str) -> Optional[str]:
    """The args blob between a layer URL's two ``ARGS_KEY`` markers.

    ``None`` when the URL carries no blob at all. Any other number of markers
    than two is a malformed URL.
    """
    if ARGS_KEY not in dataset:
        return None
    parts = dataset.split(ARGS_KEY)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid dataset format. Expected two occurrences of {ARGS_KEY}. found {len(parts)} {dataset}"
        )
    return parts[1]
