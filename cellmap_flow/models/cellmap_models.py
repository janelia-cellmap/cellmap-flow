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
    model_name: Optional[str] = None
    model_type: Optional[str] = Field(None, description="UNet or DenseNet121")
    framework: Optional[str] = Field(None, description="MONAI or PyTorch")
    spatial_dims: Optional[int] = Field(None, description="2 or 3")
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    iteration: Optional[int] = None
    input_voxel_size: Optional[List[int]] = Field(
        None, description="Comma-separated values, e.g., 8,8,8"
    )
    output_voxel_size: Optional[List[int]] = Field(
        None, description="Comma-separated values, e.g., 8,8,8"
    )
    channels_names: Optional[List[str]] = Field(
        None, description="Comma-separated values, e.g., 'CT, PET'"
    )
    input_shape: Optional[List[int]] = Field(
        None, description="Comma-separated values, e.g., 1,1,96,96,96"
    )
    output_shape: Optional[List[int]] = Field(
        None, description="Comma-separated values, e.g., 1,2,96,96,96"
    )
    author: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = "1.0.0"


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

import cellmap_flow.globals as g
from cellmap_flow.utils.bsub_utils import kill_jobs
def update_run_models(names : List[str]):
    to_be_killed = [g.cellmap_models_running[k] for k in g.cellmap_models_running if k not in names]
    kill_jobs(to_be_killed)
    to_be_submitted_models = {}

    for _,group in g.model_catalog.items():
        for k,v in group:
            if k in names and k not in g.cellmap_models_running:
                to_be_submitted_models[k]=v

    g.dataset_path
    


