jobs = []

models_config = []

servers = []

raw = None


from cellmap_flow.norm.input_normalize import MinMaxNormalizer, get_input_normalizers
from cellmap_flow.post.postprocessors import DefaultPostprocessor, get_postprocessors_list

input_norms = [MinMaxNormalizer()]
postprocess = [DefaultPostprocessor()]

viewer = None

dataset_path = None


from cellmap_flow.models.model_yaml import load_model_paths

import os

model_catalog = load_model_paths(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), os.pardir, "models", "models.yaml")
    )
)
input_norms_functions = get_input_normalizers()
output_postprocessors_functions = get_postprocessors_list()

queue = "gpu_h100"
charge_group = "cellmap"
