"""This process's normalization + postprocessing chain: ``process_chain()``.

One chain per process. In the dashboard it is the chain the user last
submitted, which the finetune manifest and the exported YAML read as
``spec``. In an inference server or a blockwise worker it is the fallback
for a layer or caller that brings no chain of its own: always empty for a
server started from the CLI, and set by the user scripts that call
``process_chunk`` themselves (ImageDataInterface's docstring says they rely
on it).

- ``input_norms`` and ``postprocess`` are the live op instances, which can
  hold state (SimpleBlockwiseMerger's equivalences), so they are replaced
  only by ``set()`` and never rebuilt from the spec behind a caller's back.
- ``input_norm_config`` and ``postprocess_config`` are the steps as the
  dashboard received them; ``spec`` reads them.
- ``set(spec, built=None)`` is the one way to change the chain.
"""

from cellmap_flow.pipeline_spec import PipelineSpec, chain_output_dtype, normalize_steps


class ProcessChain:
    """The chain; see the module docstring."""

    __slots__ = ("input_norms", "postprocess", "input_norm_config", "postprocess_config")

    def __init__(self):
        self.input_norms = []
        self.postprocess = []
        # The chain's steps as the dashboard received them, which the
        # finetune submit/restart flow hands the trainer so it normalizes
        # as inference does. Written only by set(); read them through spec,
        # which falls back to the live instances when these are empty (a
        # script that assigns input_norms or postprocess itself, as the
        # deprecated g.input_norms = [...] does, leaves them untouched).
        self.input_norm_config = {}
        self.postprocess_config = {}

    @property
    def spec(self) -> PipelineSpec:
        """The chain currently configured, as data. Derived on every read.

        Per chain: the ``*_config`` steps when set, otherwise the live
        instances' to_dict(). Never rebuilds the live instances, which can
        hold state (SimpleBlockwiseMerger's equivalences).
        """
        return PipelineSpec(
            _configured_steps(self.input_norm_config, self.input_norms),
            _configured_steps(self.postprocess_config, self.postprocess),
        )

    def set(self, spec: PipelineSpec, built=None) -> None:
        """Replace the configured chain: both live chains and both configs.

        ``built`` is the ``(input_norms, postprocess)`` instances for
        ``spec`` when the caller already has them; otherwise they are built
        here. Everything is built before anything is assigned, so a chain
        that fails to build leaves the previous one in place.
        """
        input_norms, postprocess = spec.build() if built is None else built
        self.input_norms = list(input_norms)
        self.postprocess = list(postprocess)
        self.input_norm_config = list(spec.input_norm)
        self.postprocess_config = list(spec.postprocess)

    def output_dtype(self, model_output_dtype, postprocess=None):
        """The dtype a chain hands to the client.

        ``postprocess=None`` means this chain's; the inference server passes
        the chain of the layer being served. The last step that declares a
        dtype decides, since the steps run in order: taking the first picked
        e.g. SigmoidPostprocessor's float32 ahead of a trailing
        AffinityPostprocessor's uint64, which both advertised the wrong dtype
        in the zarr metadata (neuroglancer: "Data type not compatible with
        segmentation layer") and cast uint64 label ids through float32,
        corrupting any id above 2**24.
        """
        if postprocess is None:
            postprocess = self.postprocess
        return chain_output_dtype(postprocess, model_output_dtype)


def _chain_config(steps) -> list:
    """``[{name, **params}]`` for a live chain, skipping steps that can't say."""
    derived = []
    for step in steps or []:
        try:
            d = dict(step.to_dict())
            d.setdefault("name", type(step).__name__)
            derived.append(d)
        except Exception:
            continue
    return derived


def _configured_steps(config, live):
    """The configured steps, or the live chain's when none are configured.

    The fallback matters because a script can assign the live chain itself
    (``process_chain().input_norms = [...]``, or the deprecated
    ``g.input_norms = [...]``) without touching the config; ``spec`` would
    otherwise read as empty, and a finetune manifest written from it would
    train on unnormalized input.
    """
    if config:
        return normalize_steps(config)
    return _chain_config(live)


_current = ProcessChain()


def process_chain() -> ProcessChain:
    """This process's chain.

    Read it at call time, never into a module global: tests swap in a fresh
    chain per test, and a copy taken at import would outlive them.
    """
    return _current
