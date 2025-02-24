models_config = []

servers = []

raw = None

from cellmap_flow.norm.input_normalize import MinMaxNormalizer
from cellmap_flow.post.postprocessors import DefaultPostprocessor

input_norms = [MinMaxNormalizer()]
postprocess = [DefaultPostprocessor()]

viewer = None

dataset_path = None

models_host = {}