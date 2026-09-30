"""``HuggingFaceModelConfig``: a cellmap_models export on the Hugging Face Hub.

It is downloaded, then built as a ``CellMapModelConfig``. ``to_dict()``
adds the repo's ``metadata.json`` for the pipeline builder; building a
launch command never downloads it.
"""

from cellmap_flow.models.configs.base import ModelConfig
from cellmap_flow.models.configs.cellmap import CellMapModelConfig
from cellmap_flow.utils.serialize_config import Config


class HuggingFaceModelConfig(ModelConfig):
    """Configuration class for a Hugging Face model."""

    cli_name = "huggingface"

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

    def _load_metadata(self):
        """Load metadata.json from the HuggingFace repo (cached after first call)."""
        if self._metadata is not None:
            return self._metadata
        try:
            from huggingface_hub import hf_hub_download
            import json
            path = hf_hub_download(self.repo, "metadata.json", revision=self.revision)
            with open(path) as f:
                self._metadata = json.load(f)
        except Exception:
            self._metadata = {}
        return self._metadata

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
        from cellmap_models.model_export.cellmap_model import get_huggingface_model

        cellmap_model = get_huggingface_model(self.repo, self.revision)
        config = CellMapModelConfig(folder_path=cellmap_model.folder_path)._get_config()
        return config

    def to_dict(self):
        """Export configuration for use with build_model_from_entry."""
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
