"""``MiaModelConfig``: a model trained with mia-train, the training framework
of HHMI's MIA team (the ``ai-hhmi`` organization on Hugging Face), served as
the cellmap-models export it is converted into.

Its source is a Hugging Face repo (``repo``, at ``revision``) or a folder
(``path``) holding the model's ``config.json`` and ``model.safetensors``
(cellmap_models' ``MIA_FILES``). What the network is, which of its kinds
cellmap-flow can serve, and how it is exported are cellmap_models' to know
(``model_export.mia_model``), not this module's. On its first build the model
is converted once, with ``export_mia_model``, into a cellmap-models export
(model.ts, metadata.json) in a cache folder keyed by the repo's commit or by
the folder's files, and by the kind of device it was traced on
(``export_dir``); from then on that folder is served exactly as any export
is (``CellMapModelConfig``), and finetuned as one.

The conversion imports mia-train, which only the ``mia`` pixi environment
has (the type's ``default_env``), in a Python process of its own
(``export_in_subprocess``): mia-train's top-level packages (``models``,
``layers``, ``distributed``...) stay imported once imported, and its
``distributed`` breaks Dask's in the same process. It traces on the device
of the job that first serves the model: on CUDA in bf16, else on the CPU in
fp32, each cached apart, so a server serves the export traced on its own
kind of device. Two servers starting at once each convert into a folder of
their own beside the cache folder and rename it into place: the first
rename wins and the other copy is dropped, so no server reads a
half-written export.

Nothing the dashboard asks (declared_geometry, declared_input_norm,
declared_affinity_offsets, declared_output_class, finetune_modes,
finetune_refusal, to_dict) downloads the weights or converts anything: each
reads the cached export once a server has made it (the cache is in the home
folder the dashboard and its jobs share). Before that, all but to_dict answer
from the metadata the export will have, which cellmap_models'
``mia_metadata`` makes from the repo's config.json and model card alone
(small files, which resolving the model has fetched already), so that Submit
sets the input normalization the model was trained on, and the
postprocessing its output wants, before its server has converted it.
"""

import hashlib
import importlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from cellmap_flow.models.configs.base import Config, ModelConfig
from cellmap_flow.models.configs.cellmap import CellMapModelConfig
from cellmap_flow.models.configs.huggingface import DISPLAY_KEYS
from cellmap_flow.models.resolve import EXPORT_METADATA, EXPORT_MODEL

logger = logging.getLogger(__name__)

# What knows mia-train's networks, in cellmap-models.
MIA_MODULE = "cellmap_models.model_export.mia_model"
# The pixi environment with mia-train in it (pixi.toml), the type's default.
MIA_ENV = "mia"
# Where the exports are cached: this variable's folder, else
# ~/.cellmap_flow/mia_exports. One export is the size of the model's
# weights (a ViT-L's is over a GB), more with a model.pt2.
MIA_CACHE_ENV = "CELLMAP_FLOW_MIA_CACHE"
# A repo's files downloaded beside its MIA_FILES, when it has them: its model
# card and licence, which the export's model card names.
CARD_FILES = ("README.md", "LICENSE.md")


def mia_model():
    """cellmap_models' ``model_export.mia_model``.

    Raises:
        ImportError: the cellmap_models installed has none.
    """
    try:
        return importlib.import_module(MIA_MODULE)
    except ImportError as e:
        raise ImportError(
            f"A model trained with mia-train needs a cellmap-models that has {MIA_MODULE} ({e})"
        ) from e


def cache_root() -> Path:
    """``CELLMAP_FLOW_MIA_CACHE``, else ``~/.cellmap_flow/mia_exports``."""
    override = os.environ.get(MIA_CACHE_ENV)
    return Path(override).expanduser() if override else Path.home() / ".cellmap_flow" / "mia_exports"


# The kinds of device an export is traced on, the one a peek at the cache
# (MiaModelConfig._cached) prefers first.
DEVICE_KINDS = ("cuda", "cpu")


def device_kind() -> str:
    """The kind of device an export made here is traced on: "cuda" where a
    CUDA GPU is (cellmap_models traces in bf16 there), else "cpu" (in fp32;
    an Apple GPU serves that one)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def model_dir(repo: Optional[str] = None, commit: Optional[str] = None, path: Optional[str] = None) -> Path:
    """The cache folder of the model of ``repo`` at ``commit`` (a full commit
    SHA), or of the folder ``path``, which holds its exports by device kind:
    ``<org>--<repo>/<commit>``, or ``local/<folder name>-<fingerprint>`` with
    the fingerprint of the folder's path and its MIA_FILES' sizes and times,
    so that new weights in the same folder are converted again."""
    if path is None:
        return cache_root() / repo.replace("/", "--") / commit
    fingerprint = hashlib.sha1(os.path.abspath(path).encode())
    for name in mia_model().MIA_FILES:
        stat = os.stat(os.path.join(path, name))
        fingerprint.update(f"|{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return cache_root() / "local" / f"{os.path.basename(os.path.abspath(path))}-{fingerprint.hexdigest()[:12]}"


def export_dir(repo: Optional[str] = None, commit: Optional[str] = None, path: Optional[str] = None,
               device: Optional[str] = None) -> Path:
    """The cache folder of the export of the model (``model_dir``) traced on
    ``device``, a kind of DEVICE_KINDS (this machine's, ``device_kind``, by
    default)."""
    return model_dir(repo, commit, path) / (device or device_kind())


def is_export(folder) -> bool:
    """Whether ``folder`` holds a finished export (its metadata.json and model.ts)."""
    return os.path.isfile(os.path.join(folder, EXPORT_METADATA)) and os.path.isfile(os.path.join(folder, EXPORT_MODEL))


# What the conversion's own Python runs: cellmap_models' export, and nothing
# of cellmap-flow's, so that no package of the names mia-train's are is
# imported before it.
_EXPORT_PROGRAM = (
    "import sys\n"
    "from cellmap_models.model_export.mia_model import export_mia_model\n"
    "export_mia_model(sys.argv[1], sys.argv[2])\n"
)


def export_in_subprocess(source, out) -> None:
    """``export_mia_model(source, out)`` in a Python process of its own, this
    process's python (``-P``: not the working directory's modules). Its log
    is this process's; a failure is a RuntimeError pointing there.

    Never in the process that serves: mia-train's top-level packages stay
    imported once its model is loaded, and its ``distributed`` breaks Dask's
    there; cellmap_models' loader refuses besides when a package named
    ``models``, ``layers``, ``utils`` or ``distributed`` is imported already.
    """
    result = subprocess.run([sys.executable, "-P", "-c", _EXPORT_PROGRAM, str(source), str(out)])
    if result.returncode != 0:
        raise RuntimeError(
            f"Converting the mia-train model {source} into a cellmap-models export failed (exit status "
            f"{result.returncode}); its traceback is just above in this log"
        )


def convert_once(source, target) -> None:
    """Export the mia-train model at ``source`` (a folder with its MIA_FILES)
    into ``target``, unless another process does first, in a process of its
    own (``export_in_subprocess``).

    It is exported into a new folder beside ``target`` and renamed into
    place, which is atomic: a process that finds ``target`` finds the whole
    export. When another rename got there first, that export is kept and
    this one dropped.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    work = tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.converting-")
    try:
        # A folder of its own inside work, which the exporter creates.
        out = os.path.join(work, "export")
        logger.info(f"Converting the mia-train model {source} into a cellmap-models export in {target} (once)")
        export_in_subprocess(source, out)
        if not is_export(out):
            raise RuntimeError(f"export_mia_model wrote no {EXPORT_METADATA} and {EXPORT_MODEL} into {out}")
        try:
            os.rename(out, target)
        except OSError:
            if not is_export(target):
                raise
            logger.info(f"{target} was converted by another process meanwhile; serving that one")
    finally:
        shutil.rmtree(work, ignore_errors=True)


class MiaModelConfig(ModelConfig):
    """A model trained with mia-train: a Hugging Face ``repo`` (at
    ``revision``) or a local ``path``, one of the two."""

    cli_name = "mia"

    finetunable = True

    default_env = MIA_ENV

    # A YAML's relative path is from the YAML's folder.
    config_relative_paths = ("path",)

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
        # either, what the export will declare (_planned), False when that
        # could not be read.
        self._export = None
        self._peeked = None
        self._plan = None

    # --- the export ---------------------------------------------------------------

    def export_folder(self) -> str:
        """The cellmap-models export the model is served from, converted now
        when it is not cached yet. A repo's snapshot is downloaded first (its
        MIA_FILES and CARD_FILES only), which also says the commit its export
        is cached under; with the Hub out of reach, the snapshot cached for
        ``revision`` is used."""
        if self.path:
            source, target = self.path, export_dir(path=self.path)
        else:
            from huggingface_hub import snapshot_download

            files = [*mia_model().MIA_FILES, *CARD_FILES]
            source = snapshot_download(self.repo, revision=self.revision, allow_patterns=files)
            # The snapshot's folder is its commit's SHA.
            target = export_dir(self.repo, os.path.basename(os.path.normpath(source)))
        if not is_export(target):
            convert_once(source, target)
        return str(target)

    def _served(self) -> CellMapModelConfig:
        """The export as a CellMapModelConfig, converted first when it must be."""
        if self._export is None:
            self._export = CellMapModelConfig(folder_path=self.export_folder(), name=self.name, scale=self.scale)
        return self._export

    def _cached_folder(self) -> Optional[Path]:
        """The cache folder of the export, as far as is known without the
        network or torch: a repo's commit is the one Hugging Face's local
        cache has for ``revision``; of the exports of each device kind, the
        first DEVICE_KINDS has (their metadata says the same). None when
        there is none, or the folder's files cannot be read."""
        try:
            if self.path:
                models = model_dir(path=self.path)
            else:
                from huggingface_hub import try_to_load_from_cache

                found = try_to_load_from_cache(self.repo, "config.json", revision=self.revision)
                if not isinstance(found, str):
                    return None
                models = model_dir(self.repo, Path(found).parent.name)
        except (ImportError, OSError, ValueError):
            return None
        return next((models / kind for kind in DEVICE_KINDS if is_export(models / kind)), None)

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
        return self._peeked

    def _planned(self) -> Optional[CellMapModelConfig]:
        """What the export will declare, before there is one: a
        CellMapModelConfig of cellmap_models' ``mia_metadata`` of the repo (at
        ``revision``) or folder, the metadata export_mia_model will write,
        made from its config.json and model card alone (of a repo, only those
        are fetched, never its weights). Read once; None when it cannot be (a
        cellmap_models without mia_metadata, the Hub out of reach with
        nothing cached), which is logged once."""
        if self._plan is None:
            try:
                metadata = mia_model().mia_metadata(self.repo or self.path, revision=self.revision)
                self._plan = CellMapModelConfig.from_metadata(metadata, self.name, self.scale)
            except Exception as e:  # the dashboard's questions never raise
                logger.warning(f"What {self.name} declares is not known before a server has converted it: {e}")
                self._plan = False
        return self._plan or None

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

    def finetune_modes(self):
        """The export's, or the one it will be (_declared): both, unless its
        output cannot be trained through; both when neither is known (the job
        says which it takes, as for a Hugging Face repo that cannot be
        listed)."""
        declared = self._declared()
        return declared.finetune_modes() if declared is not None else ("lora", "full")

    def finetune_refusal(self):
        """The export's, or the one it will be (_declared); None when neither is known."""
        declared = self._declared()
        return declared.finetune_refusal() if declared is not None else None

    def declared_geometry(self):
        """The export's, or the one it will be (_declared); None when neither is known."""
        declared = self._declared()
        return declared.declared_geometry() if declared is not None else None

    def declared_input_norm(self, data_range=None):
        """The export's, or the one it will be (_declared): the range its
        config.json's intensity_range says it read in training; None when
        neither is known."""
        declared = self._declared()
        return declared.declared_input_norm(data_range) if declared is not None else None

    def declared_affinity_offsets(self):
        """The export's, or the one it will be (_declared); None when neither is known."""
        declared = self._declared()
        return declared.declared_affinity_offsets() if declared is not None else None

    def declared_output_class(self):
        """The export's, or the one it will be (_declared): what cellmap_models
        declares the output is (an affinity model's affinities, a semantic
        model's uint8 probabilities x 255); None when neither is known."""
        declared = self._declared()
        return declared.declared_output_class() if declared is not None else None

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
