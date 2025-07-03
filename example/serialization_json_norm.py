#%%
from cellmap_flow.utils.serilization_utils import get_process_dataset, serialize_norms_posts_to_json
from cellmap_flow.norm.input_normalize import MinMaxNormalizer,LambdaNormalizer
from cellmap_flow.post.postprocessors import ThresholdPostprocessor
# %%
norms = [MinMaxNormalizer(), LambdaNormalizer("x*2-1")]
posts = [ThresholdPostprocessor(threshold=0.5)]
# %%
json_data = serialize_norms_posts_to_json(norms, posts)
# %%
print(json_data)
# %%
input_norm_fns, postprocess_fns = get_process_dataset(json_data)
# %%
input_norm_fns
# %%
from cellmap_flow.utils.config_utils import load_config
# %%
config = load_config("sal_1_mito.yaml")
# %%
json_data = config["json_data"]
# %%
input_norm_fns, postprocess_fns = get_process_dataset(json_data)
# %%
type(json_data)
# %%
input_norm_fns