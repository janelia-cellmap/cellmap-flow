from cellmap_flow.norm.input_normalize import get_input_normalizers

# %%
get_input_normalizers()


import cellmap_flow.norm.input_normalize
from importlib import reload

reload(cellmap_flow.norm.input_normalize)
from cellmap_flow.norm.input_normalize import (
    MinMaxNormalizer,
    NormalizationMethods,
    get_normalization,
    InputNormalizer,
)

# %%
NormalizationMethods
# %%
dic = {"name": "min_max", "min_value": 0, "max_value": 1}
# %%
get_normalization(dic)


# %%
class TINput(InputNormalizer):
    @classmethod
    def name(cls):
        return "tin"


# %%
NormalizationMethods = [f.name() for f in InputNormalizer.__subclasses__()]
# %%
NormalizationMethods
# %%
get_normalization({"name": "tin"})
# %%
NormalizationMethods
