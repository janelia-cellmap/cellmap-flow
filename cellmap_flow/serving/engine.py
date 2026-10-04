"""Which inference server a process runs: chunkmirage's or the Flask one.

``CELLMAP_FLOW_ENGINE`` is ``chunkmirage`` (the default) or ``flask``. Every
place that starts a server (``cellmap_flow serve``, ``infer --server-check``
and the finetune loop) builds it with ``make_server``, and both take the
same arguments.

A model can run in an environment of its own (its ``env``: a pixi
environment, an alias or a path), which may not have chunkmirage. Unless
chunkmirage was asked for, such a server is the Flask one, with a warning;
the dashboard draws either server's layers.
"""

import logging
import os

logger = logging.getLogger(__name__)

ENGINE_ENV = "CELLMAP_FLOW_ENGINE"
ENGINES = ("flask", "chunkmirage")
DEFAULT_ENGINE = "chunkmirage"


def engine_name(name=None) -> str:
    """``name``, else ``CELLMAP_FLOW_ENGINE``, else the default; refused if unknown."""
    name = (name or os.environ.get(ENGINE_ENV) or DEFAULT_ENGINE).strip().lower()
    if name not in ENGINES:
        raise ValueError(f"{ENGINE_ENV} must be one of {', '.join(ENGINES)}, got {name!r}")
    return name


def make_server(dataset_name, model_config, *, engine=None, **kwargs):
    """The server for ``model_config`` over ``dataset_name``; ``kwargs`` are
    CellMapFlowServer's (restart_callback, restart_token, resample)."""
    if engine_name(engine) == "chunkmirage":
        try:
            from cellmap_flow.serving.chunkmirage_server import ChunkmirageServer
        except ImportError as e:
            if engine or os.environ.get(ENGINE_ENV):
                raise
            logger.warning(f"Serving with the Flask server: chunkmirage cannot be imported here ({e})")
        else:
            return ChunkmirageServer(dataset_name, model_config, **kwargs)
    from cellmap_flow.server import CellMapFlowServer

    return CellMapFlowServer(dataset_name, model_config, **kwargs)


def check_server(server):
    """Compute one chunk the way a request would: ``infer --server-check``."""
    if hasattr(server, "read_chunk"):
        return server.read_chunk((0, 0, 0))
    return server._chunk_impl(None, None, 2, 2, 2)
