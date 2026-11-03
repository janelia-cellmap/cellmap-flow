"""``cellmap_flow max-block-size``: probe the biggest block size a model's
forward pass fits on this process's GPU, and write the result as JSON.

This is what the dashboard submits as a short job to the requested queue
(``dashboard.routes.blockwise``), not something run in the dashboard's own
process: ``max_block_size.probe_max_block_size`` needs a real GPU of the
queue's type to measure real CUDA memory against.
"""

import json
import logging
import sys

import click

from cellmap_flow.config.yaml import ConfigError
from cellmap_flow.models import registry

logger = logging.getLogger(__name__)

# The entry keys a block size pick rewrites (the builder's model node,
# routes.blockwise _sized_to_block): set aside to get at the model's own tile.
_SIZE_KEYS = ("input_size", "output_size")


def _native_config(entry):
    """The model built without the entry's ``input_size``/``output_size``,
    or None when the entry sets none, or the model does not build without
    them.

    A pick from an earlier probe is written into the model node's sizes, so
    the entry's output_size may be a block size already (392, say, for a 56
    tile). Probing multiples of that would offer 392 and 784 and nothing
    finer, and warm up on a 392 block that may not fit the GPU asked about
    now. Without them the type reads its own tile (a fly checkpoint's from
    the files beside it, else from the network), and the candidates are the
    fine-grained ones whichever block the node holds."""
    stripped = {key: value for key, value in entry.items() if key not in _SIZE_KEYS}
    if stripped == entry:
        return None
    try:
        return registry.build_model(stripped, entry.get("name"))
    except Exception as e:
        logger.info(f"Could not build the model without the entry's sizes ({e}); probing multiples of the entry's")
        return None


@click.command(name="max-block-size")
@click.option(
    "--model", "model_json", required=True,
    help='The model entry (ModelConfig.launch_entry) as JSON, e.g. '
    '\'{"type": "fly", "checkpoint_path": "/path/to/model_checkpoint_60000"}\'',
)
@click.option("--out", "out_path", required=True, type=click.Path(), help="Where to write the result JSON")
@click.option(
    "--queue", default=None,
    help="The GPU queue this runs on; given, the result is also put in the max-block-size "
    "cache (~/.cellmap_flow/max_block_size_cache.json) under this model and queue, so the "
    "dashboard finds it even if it stopped waiting for this job",
)
@click.option(
    "--max-multiple", default=8, type=int,
    help="Stop after this many multiples of the model's native output tile (default: 8)",
)
def max_block_size_cli(model_json, out_path, queue, max_multiple):
    """Probe the largest multiple of the model's native output tile that
    fits a single forward pass on this process's GPU, writing
    ``MaxBlockSizeResult.to_dict()`` as JSON to ``--out``.

    \b
      cellmap_flow max-block-size --model '{"type": "fly", "checkpoint_path": "..."}' \\
          --out result.json --queue gpu_h100
    """
    from cellmap_flow.blockwise import max_block_size_cache
    from cellmap_flow.blockwise.max_block_size import probe_max_block_size

    try:
        entry = json.loads(model_json)
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"not JSON ({e})", param_hint="'--model'")
    if not isinstance(entry, dict):
        raise click.BadParameter(f"not a model entry: {model_json}", param_hint="'--model'")
    try:
        model_config = registry.build_model(entry, entry.get("name"))
    except ConfigError as e:
        raise click.BadParameter(str(e), param_hint="'--model'")

    probed = _native_config(entry)
    if probed is not None:
        logger.info(
            f"Probing multiples of the model's own tile, not the entry's "
            f"input_size {entry.get('input_size')} / output_size {entry.get('output_size')}"
        )
    try:
        result = probe_max_block_size(probed if probed is not None else model_config, max_multiple=max_multiple)
    except Exception:
        logger.exception(f"Max block size probe failed for {model_json}")
        sys.exit(1)

    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f)
    if queue:
        try:
            max_block_size_cache.store(model_config, queue, result)
        except Exception as e:  # the file written above is the answer; the cache is a convenience
            logger.warning(f"Could not cache the result for {queue}: {e}")
    best = result.best
    if best is None:
        click.echo(f"Even the native block size did not fit on {result.gpu_name}; see {out_path}")
    else:
        click.echo(
            f"Largest block size on {result.gpu_name}: x{best.multiple} {best.block_shape} "
            f"({best.peak_bytes / (1024 ** 3):.2f} GiB); see {out_path}"
        )
