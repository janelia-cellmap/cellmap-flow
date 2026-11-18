"""``MiaModelConfig``: a model trained with mia-train, the training framework
of HHMI's MIA team (the ``ai-hhmi`` organization on Hugging Face), served as
the cellmap-models export it is converted into.

Its source is a Hugging Face repo (``repo``, at ``revision``) or a folder
(``path``) holding the model's ``config.json`` and ``model.safetensors``.
What the network is, which of its kinds cellmap-flow can serve, how it is
exported, and how its exports are cached are cellmap_models' to know
(``model_export.mia_model`` and ``model_export.mia_cache``), not this
module's. On its first build the model is converted once into a
cellmap-models export in cellmap-flow's cache folder (``cache_root``), by
``mia_cache.cached_export``, in a Python process of its own in the ``mia``
pixi environment, which has mia-train (the type's ``default_env``); from
then on that folder is served exactly as any export is
(``CellMapModelConfig``), and finetuned as one.

Nothing the dashboard or blockwise asks (declared_geometry,
declared_input_norm, declared_affinity_offsets, declared_class_ids, declared_postprocess,
declared_overlap, finetune_modes, finetune_refusal, planned_metadata,
to_dict) downloads the weights or converts anything: each
reads the cached export once a server has made it (the cache is in the home
folder the dashboard and its jobs share). Before that, all but to_dict answer
from the metadata the export will have, which cellmap_models'
``mia_metadata`` makes from the repo's config.json and model card alone
(small files, which resolving the model has fetched already), so that Submit
sets the input normalization the model was trained on before its server
has converted it. While that cannot be read either (the Hub out of reach,
with nothing cached), they say nothing is declared and that the model cannot
be finetuned yet, and it is read again a minute on (``PLAN_RETRY_SECONDS``).
"""

import importlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

from cellmap_flow.models.configs.base import Config, ModelConfig
from cellmap_flow.models.configs.cellmap import CellMapModelConfig
from cellmap_flow.models.configs.huggingface import DISPLAY_KEYS, HuggingFaceModelConfig
# cellmap_models' model_export.mia_model, which knows mia-train's networks,
# has one loader, resolve's, which tells the references that are such models.
from cellmap_flow.models.resolve import mia_model

logger = logging.getLogger(__name__)

# The pixi environment with mia-train in it (pixi.toml), the type's default.
MIA_ENV = "mia"
# Where the exports are cached: this variable's folder, else
# ~/.cellmap_flow/mia_exports. One export is the size of the model's
# weights (a ViT-L's is over a GB), more with a model.pt2.
MIA_CACHE_ENV = "CELLMAP_FLOW_MIA_CACHE"


def cache_root() -> Path:
    """``CELLMAP_FLOW_MIA_CACHE``, else ``~/.cellmap_flow/mia_exports``."""
    override = os.environ.get(MIA_CACHE_ENV)
    return Path(override).expanduser() if override else Path.home() / ".cellmap_flow" / "mia_exports"


def mia_cache():
    """cellmap_models' ``model_export.mia_cache``, which converts a model once
    into the cache and finds its export again; imported now.

    Raises:
        ImportError: the cellmap_models installed has none.
    """
    return importlib.import_module("cellmap_models.model_export.mia_cache")


def _as_declared(name, unknown=None):
    """A MiaModelConfig method that answers ``name`` as the export does
    (``CellMapModelConfig``'s method of that name, with the same arguments),
    or as the export it will be does (``MiaModelConfig._declared``);
    ``unknown`` when neither is known."""

    def ask(self, *args, **kwargs):
        declared = self._declared()
        return getattr(declared, name)(*args, **kwargs) if declared is not None else unknown

    ask.__name__ = name
    ask.__doc__ = (f"The export's (``CellMapModelConfig.{name}``), or the one it will be (_declared); "
                   f"{unknown!r} when neither is known.")
    return ask


class MiaModelConfig(ModelConfig):
    """A model trained with mia-train: a Hugging Face ``repo`` (at
    ``revision``) or a local ``path``, one of the two."""

    cli_name = "mia"

    finetunable = True

    default_env = MIA_ENV

    # A YAML's relative path is from the YAML's folder.
    config_relative_paths = ("path",)

    # How long a plan (_planned) that could not be read is not asked for
    # again: as long as a Hugging Face model's metadata.json is not, for the
    # same reason (the dashboard asks on every poll, and an unreachable Hub
    # costs a timeout a try).
    PLAN_RETRY_SECONDS = HuggingFaceModelConfig.METADATA_RETRY_SECONDS

    def __init__(self, repo: str = None, path: str = None, revision: str = None, name: str = None, scale=None):
        super().__init__()
        if bool(repo) == bool(path):
            raise ValueError(
                "A mia model needs either repo (a Hugging Face org/repo) or path (a folder with the model's "
                "config.json and model.safetensors), not both or neither"
            )
        self.repo = repo or None
        self.path = os.path.abspath(os.path.expanduser(path)) if path else None
        self.revision = revision or None
        self.name = name or (self.repo.split("/")[-1] if self.repo else os.path.basename(self.path.rstrip("/")))
        self.scale = scale
        # The export served, once _served() has made sure of it; the one
        # found in the cache by _cached(), which made nothing; and, before
        # either, what the export will declare (_planned), None until it has
        # been read, with when reading it last failed.
        self._export = None
        self._peeked = None
        self._plan = None
        self._plan_failed_at = None

    # --- the export ---------------------------------------------------------------

    def export_folder(self) -> str:
        """The cellmap-models export the model is served from, in the cache
        (``cache_root``), converted now when it is not there yet
        (``mia_cache.cached_export``, which downloads a repo's files first).

        Raises:
            RuntimeError: the model must be converted and its weights could
                not be downloaded, or the conversion failed.
        """
        return mia_cache().cached_export(cache_root(), repo=self.repo, revision=self.revision, path=self.path)

    def _served(self) -> CellMapModelConfig:
        """The export as a CellMapModelConfig, converted first when it must be."""
        if self._export is None:
            self._export = CellMapModelConfig(folder_path=self.export_folder(), name=self.name, scale=self.scale)
        return self._export

    def _cached_folder(self) -> Optional[Path]:
        """The cache folder of the export, as far as is known without the
        network or torch (``mia_cache.find_export``); None when there is
        none, the folder's files cannot be read, or the cellmap_models
        installed has no cache."""
        try:
            return mia_cache().find_export(cache_root(), repo=self.repo, revision=self.revision, path=self.path)
        except (ImportError, OSError, ValueError):
            return None

    def _cached(self) -> Optional[CellMapModelConfig]:
        """The export being served, else the one cached already (a server
        made it), as a CellMapModelConfig; None before either exists. Never
        downloads or converts."""
        if self._export is not None:
            return self._export
        folder = self._cached_folder()
        if folder is None:
            return None
        if self._peeked is None or self._peeked.cellmap_model.folder_path != str(folder):
            try:
                self._peeked = CellMapModelConfig(folder_path=str(folder), name=self.name, scale=self.scale)
            except Exception as e:  # an export that cannot be read says so when it is served
                logger.debug(f"Could not read the cached export {folder}: {e}")
                return None
            mia_cache().mark_used(folder)  # a dashboard's or a trainer's install uses it too
        return self._peeked

    def _planned(self) -> Optional[CellMapModelConfig]:
        """What the export will declare, before there is one: a
        CellMapModelConfig of cellmap_models' ``mia_metadata`` of the repo (at
        ``revision``) or folder, the metadata export_mia_model will write,
        made from its config.json and model card alone (of a repo, only those
        are fetched, never its weights). Read once; None while it cannot be
        (a cellmap_models without mia_metadata, the Hub out of reach with
        nothing cached), which is logged, and read again PLAN_RETRY_SECONDS
        on: a brief Hub error hides it for a minute, not for the life of the
        config the dashboard keeps."""
        if self._plan is None and (
            self._plan_failed_at is None or time.monotonic() - self._plan_failed_at >= self.PLAN_RETRY_SECONDS
        ):
            try:
                metadata = mia_model().mia_metadata(self.repo or self.path, revision=self.revision)
                self._plan = CellMapModelConfig.from_metadata(metadata, self.name, self.scale)
            except Exception as e:  # the dashboard's questions never raise
                self._plan_failed_at = time.monotonic()
                logger.warning(f"What {self.name} declares is not known before a server has converted it ({e}); "
                               f"trying again after {self.PLAN_RETRY_SECONDS} s")
        return self._plan

    def planned_metadata(self):
        """What the export will declare: cellmap_models' ModelMetadata that
        ``mia_metadata`` made (_planned), read as _planned reads it; None
        while it cannot be read. Until a server has converted the model, what
        this config answers is answered from it, so while it is None those
        answers are no answers (finetune_modes none, declared_input_norm
        None), and a list of models (hf_catalog) keeps none of them."""
        plan = self._planned()
        return plan.cellmap_model.metadata if plan is not None else None

    def _declared(self) -> Optional[CellMapModelConfig]:
        """What answers what the model declares: its export, served or
        cached (_cached), else what the export will be (_planned); None when
        neither is known."""
        export = self._cached()
        return export if export is not None else self._planned()

    @property
    def cellmap_model(self):
        """The export's CellmapModel (the trainer's, finetune.model_loading,
        once the config is built), or None while there is none."""
        export = self._cached()
        return export.cellmap_model if export is not None else None

    # --- what the rest of cellmap-flow asks ------------------------------------------

    def _get_config(self) -> Config:
        return self._served()._get_config()

    def runs_at_native_voxel_size(self) -> bool:
        """The export's (``CellMapModelConfig.runs_at_native_voxel_size``), or
        the one it will be (_declared); converted first when neither is
        known, as the server asks right before it builds the model."""
        declared = self._declared()
        return (declared if declared is not None else self._served()).runs_at_native_voxel_size()

    def finetune_refusal(self):
        """The export's, or the one it will be (_declared). While neither is
        known, that the model cannot be finetuned until it is: its output may
        be a whole number (a semantic model's uint8 probabilities x 255),
        which no gradient goes through, so a guess would offer a training
        that fails on the GPU."""
        declared = self._declared()
        if declared is not None:
            return declared.finetune_refusal()
        return (
            f"{self.name} cannot be finetuned until it is known what its export will declare: that could not be "
            "read from its config.json (the log says why), and no server has converted it yet. Starting its "
            f"server converts it; else it is read again in {self.PLAN_RETRY_SECONDS} s."
        )

    # What the model declares is the export's answer, or that of the export
    # it will be (_as_declared); never converting, so that blockwise refuses
    # a model with an overlap before anything builds it (a first build
    # converts it: minutes on a CPU).
    # Both modes, unless its output cannot be trained through; none while
    # neither is known, as finetune_refusal says.
    finetune_modes = _as_declared("finetune_modes", unknown=())
    # One number of output voxels, or one per axis, as its metadata gives it.
    declared_overlap = _as_declared("declared_overlap")
    declared_geometry = _as_declared("declared_geometry")
    # The range its config.json's intensity_range says it read in training.
    declared_input_norm = _as_declared("declared_input_norm")
    declared_affinity_offsets = _as_declared("declared_affinity_offsets")
    # A semantic model's classes' ids in CellMap's annotations.
    declared_class_ids = _as_declared("declared_class_ids")

    def declared_postprocess(self):
        """The export's, or the one it will be (_declared): the step its
        metadata declares is applied after the model. cellmap_models declares
        a mia-train export's activation in the export, so none; [] when
        neither is known."""
        declared = self._declared()
        return declared.declared_postprocess() if declared is not None else []

    def _launch_params(self) -> dict:
        # Not to_dict(): its display fields are no constructor's arguments.
        params = {"repo": self.repo, "path": self.path, "revision": self.revision}
        return self._with_name_scale({key: value for key, value in params.items() if value is not None})

    def to_dict(self):
        """This config as a model entry, which ``registry.build_model``
        rebuilds; with the cached export's metadata, for the pipeline builder
        to show, as a Hugging Face model's has its repo's."""
        result = {"type": "mia", **self._launch_params()}
        export = self._cached()
        if export is not None:
            written = export._written_metadata()
            result.update({key: written[key] for key in DISPLAY_KEYS if key in written})
        return result
