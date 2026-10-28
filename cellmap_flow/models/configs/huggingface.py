"""``HuggingFaceModelConfig``: a cellmap_models export on the Hugging Face Hub.

It is the export at the repository's root, downloaded (``download_export``),
then built as a ``CellMapModelConfig``. ``to_dict()`` adds the repo's
``metadata.json`` for the pipeline builder; building a launch command never
downloads it.
"""

import logging
import time
from types import SimpleNamespace

from cellmap_flow.models.configs.base import Config, ModelConfig
from cellmap_flow.models.configs.cellmap import CellMapModelConfig, metadata_geometry
from cellmap_flow.serving.probe import metadata_input_norm

logger = logging.getLogger(__name__)


def download_export(repo, revision=None):
    """The cellmap_models export at the root of the Hugging Face repository
    ``repo``, downloaded: its CellmapModel.

    The root's files only. A model a server published
    (dashboard/routes/finetune/publish.py) has its ONNX web export in a
    folder beside them, a model of its own (``<repo>/<folder>``, type
    onnx) whose files are of no use here and can be gigabytes.
    """
    from cellmap_models.model_export.cellmap_model import CellmapModel
    from huggingface_hub import snapshot_download

    return CellmapModel(snapshot_download(repo, revision=revision, ignore_patterns="*/*"))


class HuggingFaceModelConfig(ModelConfig):
    """Configuration class for a Hugging Face model."""

    cli_name = "huggingface"

    finetunable = True

    # How long a failed metadata.json download is remembered before the next
    # to_dict() tries again. Not zero: the pipeline builder and the model
    # list call to_dict() on every request, and an unreachable Hub costs
    # hf_hub_download's timeout per model per try.
    METADATA_RETRY_SECONDS = 60

    def __init__(self, repo, revision=None, name=None, scale=None):
        super().__init__()
        self.repo = repo
        self.revision = revision
        if name is None:
            # Use repo name as default
            name = repo.split("/")[-1]
        self.name = name
        self.scale = scale
        self._metadata = None
        self._metadata_failed_at = None
        self._files = None
        self._files_failed_at = None

    def _load_metadata(self):
        """The repo's metadata.json, or {} when it cannot be had.

        Once read, it is kept for the life of this config. A failure is
        remembered for METADATA_RETRY_SECONDS, so a network blip hides the
        metadata for a minute rather than for the rest of the process.
        """
        if self._metadata is not None:
            return self._metadata
        if (
            self._metadata_failed_at is not None
            and time.monotonic() - self._metadata_failed_at < self.METADATA_RETRY_SECONDS
        ):
            return {}
        try:
            from huggingface_hub import hf_hub_download
            import json
            path = hf_hub_download(self.repo, "metadata.json", revision=self.revision)
            with open(path) as f:
                self._metadata = json.load(f)
        except Exception as e:
            self._metadata_failed_at = time.monotonic()
            logger.warning(
                f"Could not read metadata.json of {self.repo} ({e}); "
                f"trying again after {self.METADATA_RETRY_SECONDS} s"
            )
            return {}
        return self._metadata

    def finetune_modes(self):
        """LoRA and full when the repo has a ``model.pt2``, full only without
        one, as for an export's folder (CellMapModelConfig.finetune_modes):
        cellmap_models' ``train()`` then hands the trainer the TorchScript,
        which takes no adapter. cellmap/jrc_mus-livers_16nm_to_8nm_mito has
        model.pt, model.ts and model.onnx: offered LoRA, its job failed at
        its start. The repo's file list is asked for, not the model, once for
        the life of this config; when the Hub cannot say, both are offered,
        and the job says which it takes."""
        if self._files is None and (
            self._files_failed_at is None
            or time.monotonic() - self._files_failed_at >= self.METADATA_RETRY_SECONDS
        ):
            from cellmap_flow.models.resolve import _hf_files

            try:
                self._files = _hf_files(self.repo, self.revision)
            except Exception as e:
                self._files_failed_at = time.monotonic()
                logger.warning(f"Could not list the files of {self.repo} ({e}); offering LoRA and full")
        if self._files is not None and "model.pt2" not in self._files:
            return ("full",)
        return ("lora", "full")

    def declared_geometry(self):
        """Its metadata.json's, as a cellmap export's (``metadata_geometry``):
        that file is downloaded, not the model."""
        metadata = self._load_metadata()
        return metadata_geometry(SimpleNamespace(**metadata)) if isinstance(metadata, dict) else None

    def declared_input_norm(self, data_range=None):
        """The input normalization the repo's metadata.json declares: its
        ``input_norm``, cellmap-flow steps, which a model a server published
        has (finetune.export_web); else a cellmap-models export's
        ``input_normalization``, the range the model reads
        (``metadata_input_norm``), for raw data spanning ``data_range``.
        Downloaded as for declared_geometry; cheap after the first ask, as
        ``_load_metadata`` is, and never raises."""
        metadata = self._load_metadata()
        if not isinstance(metadata, dict):
            return None
        return metadata.get("input_norm") or metadata_input_norm(metadata.get("input_normalization"), data_range)

    def _launch_params(self) -> dict:
        # Not to_dict(): that downloads metadata.json for the pipeline builder,
        # and launching a server needs only the constructor arguments.
        return {
            "repo": self.repo,
            "revision": self.revision,
            "name": self.name,
            "scale": self.scale,
        }

    def _get_config(self) -> Config:
        cellmap_model = download_export(self.repo, self.revision)
        config = CellMapModelConfig(folder_path=cellmap_model.folder_path)._get_config()
        return config

    def to_dict(self):
        """This config as a model entry, which ``registry.build_model`` rebuilds."""
        result = {"type": "huggingface", "repo": self.repo}
        if self.revision is not None:
            result["revision"] = self.revision
        self._with_name_scale(result)

        # Include metadata from HuggingFace model for pipeline builder display
        metadata = self._load_metadata()
        if metadata:
            for key in ("channels_names", "input_voxel_size", "output_voxel_size",
                        "inference_input_shape", "inference_output_shape",
                        "in_channels", "out_channels", "model_type", "framework",
                        "spatial_dims", "iteration", "model_name", "description"):
                if key in metadata:
                    result[key] = metadata[key]

        return result
