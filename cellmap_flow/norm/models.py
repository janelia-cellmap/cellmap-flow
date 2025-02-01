# TMP code just a mockup
from cellmap_flow.utils.data import BioModelConfig, ScriptModelConfig, DaCapoModelConfig


def get_available_models():
    return [
        BioModelConfig("model_name"),
        ScriptModelConfig("script_path"),
        DaCapoModelConfig("run_name", 1),
    ]
