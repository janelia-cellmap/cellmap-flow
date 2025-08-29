import os
import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# For demonstration of loading .onnx and .pt / .ts models:
try:
    import onnxruntime as ort
except ImportError:
    ort = None  # If onnxruntime isn't installed, set it to None

try:
    import torch
except ImportError:
    torch = None  # If torch isn't installed, set it to None


class ModelMetadata(BaseModel):
    model_name: Optional[str] = Field(None, description="Name of the model")
    model_type: Optional[str] = Field(
        None, description="Type of the model, e.g., UNet or DenseNet121"
    )
    framework: Optional[str] = Field(
        None, description="Framework used, e.g., MONAI or PyTorch"
    )
    spatial_dims: Optional[int] = Field(
        None, description="Number of spatial dimensions, e.g., 2 or 3"
    )
    in_channels: Optional[int] = Field(None, description="Number of input channels")
    out_channels: Optional[int] = Field(None, description="Number of output channels")
    iteration: Optional[int] = Field(None, description="Iteration number")
    input_voxel_size: Optional[List[int]] = Field(
        None, description="Input voxel size as comma-separated values, e.g., 8,8,8"
    )
    output_voxel_size: Optional[List[int]] = Field(
        None, description="Output voxel size as comma-separated values, e.g., 8,8,8"
    )
    channels_names: Optional[List[str]] = Field(
        None,
        description="Names of the channels as comma-separated values, e.g., 'CT, PET'",
    )
    input_shape: Optional[List[int]] = Field(
        None, description="Input shape as comma-separated values, e.g., 1,1,96,96,96"
    )
    output_shape: Optional[List[int]] = Field(
        None, description="Output shape as comma-separated values, e.g., 1,2,96,96,96"
    )
    inference_input_shape: Optional[List[int]] = Field(
        None,
        description="Inference input shape as comma-separated values, e.g., 1,1,96,96,96",
    )
    inference_output_shape: Optional[List[int]] = Field(
        None,
        description="Inference output shape as comma-separated values, e.g., 1,2,96,96,96",
    )
    author: Optional[str] = Field(None, description="Author of the model")
    description: Optional[str] = Field(None, description="Description of the model")
    version: Optional[str] = Field("1.0.0", description="Version of the model")


class CellmapModel:
    """
    Represents a single model directory.
    Lazily loads:
      - metadata.json --> pydantic ModelMetadata
      - model.onnx    --> ONNX model session (if onnxruntime is available)
      - model.pt      --> PyTorch model (if torch is available)
      - model.ts      --> TorchScript model (if torch is available)
      - README.md      --> str
    """

    def __init__(self, folder_path: str):
        self.folder_path = folder_path

        # Internal cache for lazy properties
        self._metadata: Optional[ModelMetadata] = None
        self._readme_content: Optional[str] = None

        self._onnx_model = None
        self._pt_model = None
        self._ts_model = None

    @property
    def metadata(self) -> ModelMetadata:
        """Lazy load the metadata.json file and parse it into a ModelMetadata object."""
        if self._metadata is None:
            metadata_file = os.path.join(self.folder_path, "metadata.json")
            metadata_file = os.path.normpath(metadata_file)
            with open(metadata_file, "r") as f:
                data = json.load(f)
            self._metadata = ModelMetadata(**data)
        return self._metadata

    @property
    def onnx_model(self):
        """
        If 'model.onnx' exists, lazily load it as an ONNX Runtime InferenceSession.
        Use GPU if available (requires onnxruntime-gpu installed), otherwise CPU.
        Returns None if the file doesn't exist or onnxruntime isn't installed.
        """
        if self._onnx_model is None:
            model_path = os.path.join(self.folder_path, "model.onnx")
            model_path = os.path.normpath(model_path)
            if ort is None:
                # onnxruntime is not installed
                return None

            if os.path.exists(model_path):
                # Check available execution providers
                available_providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in available_providers:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]

                self._onnx_model = ort.InferenceSession(model_path, providers=providers)
            else:
                self._onnx_model = None

        return self._onnx_model

    @property
    def pytorch_model(self):
        """
        If 'model.pt' exists, lazily load it using torch.load().
        Returns None if the file doesn't exist or PyTorch isn't installed.

        NOTE: Adjust this for how your .pt was saved (entire model vs state_dict).
        """
        if self._pt_model is None:
            if torch is None:
                # PyTorch is not installed
                return None
            pt_path = os.path.join(self.folder_path, "model.pt")
            pt_path = os.path.normpath(pt_path)
            if os.path.exists(pt_path):
                # Load the entire model object.
                # If your file only has the state_dict, you'll need to do something like:
                #   model = MyModelClass(...)  # define your model arch
                #   model.load_state_dict(torch.load(pt_path))
                #   self._pt_model = model
                # Instead of just torch.load().
                self._pt_model = torch.load(pt_path)
            else:
                self._pt_model = None
        return self._pt_model

    @property
    def ts_model(self):
        """
        If 'model.ts' exists, lazily load it using torch.jit.load().
        Returns None if the file doesn't exist or PyTorch isn't installed.
        """
        if self._ts_model is None:
            if torch is None:
                # PyTorch is not installed
                return None
            ts_path = os.path.join(self.folder_path, "model.ts")
            ts_path = os.path.normpath(ts_path)
            if os.path.exists(ts_path):
                self._ts_model = torch.jit.load(ts_path)
            else:
                self._ts_model = None
        return self._ts_model

    @property
    def readme(self) -> Optional[str]:
        """
        Lazy load the README.md content if it exists, else None.
        """
        if self._readme_content is None:
            readme_file = os.path.join(self.folder_path, "README.md")
            readme_file = os.path.normpath(readme_file)
            if os.path.exists(readme_file):
                with open(readme_file, "r", encoding="utf-8") as f:
                    self._readme_content = f.read()
            else:
                self._readme_content = None
        return self._readme_content


class CellmapModels:
    """
    A container that discovers all subfolders in the given directory
    and provides them as model attributes.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._models: Dict[str, CellmapModel] = {}

        # Pre-scan subfolders for potential models
        for folder in os.listdir(root_dir):
            full_path = os.path.join(root_dir, folder)
            if os.path.isdir(full_path):
                # We assume that if there's a metadata.json, it's a model directory
                if os.path.exists(os.path.join(full_path, "metadata.json")):
                    # Store in dictionary with the folder name as the key
                    self._models[folder] = CellmapModel(full_path)

    def __getattr__(self, name: str) -> CellmapModel:
        """
        Expose subfolders as attributes by name.
        For example, if there's a subfolder 'v21_mito_attention', you can do:
            cellmap_models.v21_mito_attention.metadata
        """
        if name in self._models:
            return self._models[name]
        raise AttributeError(f"No model named '{name}' in {self.root_dir}")

    def list_models(self) -> List[str]:
        """
        Returns the list of detected model names (subfolder names
        that contain 'metadata.json').
        """
        return list(self._models.keys())
