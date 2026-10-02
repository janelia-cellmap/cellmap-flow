"""Deprecated: ``g`` stands in for state that now lives with its owners.

``cellmap_flow.globals.g`` used to hold everything a process shared. Each
part now has an owner, and ``g`` forwards every name it had to that owner,
warning with a DeprecationWarning that names the replacement:

- the launcher settings (``queue``, ``charge_group``, ``walltime``, ...,
  ``save_server_config()``): ``cellmap_flow.jobs.settings.launcher_settings()``;
- the chain (``input_norms``, ``postprocess``, their ``*_config``,
  ``pipeline_spec``, ``set_pipeline()``, ``get_output_dtype()``):
  ``cellmap_flow.process_chain.process_chain()``;
- the servers this process started (``jobs``):
  ``cellmap_flow.jobs.launch.started_jobs()``;
- the rest, the dashboard's state: ``cellmap_flow.dashboard.state.get_session()``.

``g`` and ``Flow`` go in the release after 0.3.0. Until then importing this
module still configures logging, as it always has, so the scripts that
import it keep their log lines. The owners are imported only when a name is
used, so importing this module pulls in neither the dashboard nor jobs.
"""

import warnings

from cellmap_flow.logging_setup import configure_logging

configure_logging()

_SETTINGS = "cellmap_flow.jobs.settings.launcher_settings()"
_CHAIN = "cellmap_flow.process_chain.process_chain()"
_JOBS = "cellmap_flow.jobs.launch.started_jobs()"
_SESSION = "cellmap_flow.dashboard.state.get_session()"


def _settings():
    from cellmap_flow.jobs.settings import launcher_settings

    return launcher_settings()


def _chain():
    from cellmap_flow.process_chain import process_chain

    return process_chain()


def _jobs():
    from cellmap_flow.jobs.launch import started_jobs

    return started_jobs()


def _session():
    from cellmap_flow.dashboard.state import get_session

    return get_session()


class _Forward:
    """One of g's names: what replaces it, and how to read and write it there."""

    __slots__ = ("replacement", "get", "set")

    def __init__(self, replacement, get, set=None):
        self.replacement, self.get, self.set = replacement, get, set


def _attribute(owner, owner_text, attr, writable=True):
    """``owner().<attr>``."""
    return _Forward(
        f"{owner_text}.{attr}",
        lambda: getattr(owner(), attr),
        (lambda value: setattr(owner(), attr, value)) if writable else None,
    )


def _builder(key):
    """One node list of the builder's last apply, ``builder_state[key]``."""
    def set_(value):
        session = _session()
        state = session.builder_state
        state[key] = value
        session.builder_state = state

    return _Forward(f'{_SESSION}.builder_state["{key}"]', lambda: _session().builder_state[key], set_)


def _replace_jobs(jobs):
    # In place: start_hosts appends to this list and cleanup_handler kills
    # from it, so it must stay the same list.
    _jobs()[:] = jobs


# Every name g had, and nothing else: new code uses the owners. In particular
# a setting added after g was deprecated is not added here.
_FORWARDS = {
    **{key: _attribute(_settings, _SETTINGS, key) for key in (
        "queue", "charge_group", "walltime", "cycle_gpu_queues", "nb_cores_master", "nb_cores_worker",
        "nb_workers")},
    "_server_config_cached": _attribute(_settings, _SETTINGS, "cached"),
    # A plain write, as it always was: g.input_norms = [...] leaves the
    # configured steps alone.
    **{name: _attribute(_chain, _CHAIN, name) for name in (
        "input_norms", "postprocess", "input_norm_config", "postprocess_config")},
    "pipeline_spec": _attribute(_chain, _CHAIN, "spec", writable=False),
    "jobs": _Forward(_JOBS, _jobs, _replace_jobs),
    "NEUROGLANCER_URL": _attribute(_session, _SESSION, "neuroglancer_url"),
    **{f"pipeline_{key}": _builder(key) for key in (
        "inputs", "outputs", "edges", "normalizers", "models", "postprocessors")},
    "pipeline_model_configs": _attribute(_session, _SESSION, "builder_model_configs"),
    **{name: _attribute(_session, _SESSION, name) for name in (
        "viewer", "dataset_path", "raw", "extra_layers", "shaders", "shader_controls", "models_config",
        "model_catalog", "tmp_dir", "blockwise_tasks_dir", "log_buffer", "log_clients", "bbx_generator_state",
        "review", "minio_state", "annotation_volumes", "output_sessions", "finetune_job_manager")},
}


def _warn(name, replacement):
    # stacklevel 3: past this function and g's accessor, to the line that
    # used g, which is where the fix goes.
    warnings.warn(
        f"cellmap_flow.globals.{name} is deprecated and goes in the release after 0.3.0; use {replacement}",
        DeprecationWarning,
        stacklevel=3,
    )


class Flow:
    """Deprecated: the type of ``g``; see the module docstring.

    Every attribute is forwarded to its owner on each use, never copied, so
    ``g`` and the owners cannot disagree. It stores nothing itself: a name g
    never had raises, as a misspelt one does on the dashboard's Session,
    rather than being kept where nothing reads it.
    """

    __slots__ = ()

    def __new__(cls):
        _warn("Flow()", "the owners that cellmap_flow.globals' docstring lists (Flow() returns g)")
        return g

    def __getattr__(self, name):
        forward = _FORWARDS.get(name)
        if forward is None:
            raise AttributeError(f"cellmap_flow.globals.g has no attribute {name!r}")
        _warn(f"g.{name}", forward.replacement)
        return forward.get()

    def __setattr__(self, name, value):
        forward = _FORWARDS.get(name)
        if forward is None:
            raise AttributeError(f"cellmap_flow.globals.g has no attribute {name!r}")
        if forward.set is None:
            raise AttributeError(f"cellmap_flow.globals.g.{name} cannot be set; it is derived from the chain")
        _warn(f"g.{name}", forward.replacement)
        forward.set(value)

    def __dir__(self):
        return sorted({*object.__dir__(self), *_FORWARDS})

    def __repr__(self):
        return "<cellmap_flow.globals.g, deprecated: see the cellmap_flow.globals docstring>"

    def save_server_config(self):
        _warn("g.save_server_config()", f"{_SETTINGS}.save()")
        _settings().save()

    def set_pipeline(self, spec, built=None):
        _warn("g.set_pipeline()", f"{_CHAIN}.set()")
        _chain().set(spec, built=built)

    def get_output_dtype(self, model_output_dtype, postprocess=None):
        _warn("g.get_output_dtype()", f"{_CHAIN}.output_dtype()")
        return _chain().output_dtype(model_output_dtype, postprocess)


g = object.__new__(Flow)
