"""Building the inference server a process runs, and the chunk that checks it.

Every place that starts a server (``cellmap_flow serve``, ``infer
--server-check`` and the finetune loop) builds it with ``make_server``, so
they agree on what a server is: ``serving.chunkmirage_server.ChunkmirageServer``.

A model may run in an environment of its own (its ``env``: a pixi
environment, an alias or a path). That environment needs chunkmirage, as
every environment cellmap-flow is installed in has; one without it is told
so instead of serving.
"""


def make_server(dataset_name, model_config, **kwargs):
    """The server for ``model_config`` over ``dataset_name``; ``kwargs`` are
    ChunkmirageServer's (restart_callback, stop_callback, restart_token, resample)."""
    try:
        from cellmap_flow.serving.chunkmirage_server import ChunkmirageServer
    except ImportError as e:
        raise ImportError(
            f"The inference server needs chunkmirage, which cannot be imported here ({e}). "
            "Install cellmap-flow's dependencies in the environment the model runs in "
            "(its `env`, when it has one of its own)."
        ) from e
    return ChunkmirageServer(dataset_name, model_config, **kwargs)


def check_server(server):
    """Compute one chunk the way a request would: ``infer --server-check``."""
    return server.read_chunk((0, 0, 0))
