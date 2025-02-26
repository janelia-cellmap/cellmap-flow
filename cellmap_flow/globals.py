processes = []
job_ids = []

models_config = []

servers = []

raw = None

cellmap_models_running = {} # name of the model , job_id

from cellmap_flow.norm.input_normalize import MinMaxNormalizer
from cellmap_flow.post.postprocessors import DefaultPostprocessor

input_norms = [MinMaxNormalizer()]
postprocess = [DefaultPostprocessor()]

viewer = None

dataset_path = None

models_host = {}

from cellmap_flow.models.model_yaml import load_model_paths
model_catalog = {}
# # TODO: as a parameter to the app
# model_catalog = load_model_paths(
#     "/Users/zouinkhim/Desktop/cellmap/flo/cellmap-flow/example/models.yaml"
# )
