import logging
from cellmap_flow.utils.web_utils import (
    decode_to_json,
    ARGS_KEY,
    INPUT_NORM_DICT_KEY,
    POSTPROCESS_DICT_KEY,
)
from cellmap_flow.norm.input_normalize import get_normalizations
from cellmap_flow.post.postprocessors import get_postprocessors

# from cellmap_flow.utils.web_utils import encode_to_str, decode_to_json
import json

logger = logging.getLogger(__name__)


def get_process_dataset(json_data: dict):
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    logger.info(f"json data: {json_data}")
    input_norm_fns = get_normalizations(json_data[INPUT_NORM_DICT_KEY])
    postprocess_fns = get_postprocessors(json_data[POSTPROCESS_DICT_KEY])
    return input_norm_fns, postprocess_fns


def get_process_dataset_url(dataset: str):
    if ARGS_KEY not in dataset:
        # A layer URL without the args blob means this request carries no
        # normalization and no postprocessing, and the model is about to be
        # fed raw voxel values. For a model trained on, say, [-1, 1] that is
        # not a subtle degradation -- the output is unrecognizable, and it
        # looks exactly like a model that "trained badly" rather than one
        # that is being served wrong. Returning three empty values in silence
        # is what made that indistinguishable, so say it out loud.
        logger.warning(
            "Serving WITHOUT normalization or postprocessing: the layer URL "
            f"has no {ARGS_KEY} block. Raw voxel values go to the model "
            "unmodified. If the model expects normalized input (e.g. [-1, 1]) "
            "its output will be meaningless. Re-add the layer from the "
            "dashboard so the URL carries the current Input/Postprocess "
            "configuration."
        )
        return None, [], []
    norm_data = dataset.split(ARGS_KEY)
    if len(norm_data) != 3:
        raise ValueError(
            f"Invalid dataset format. Expected two occurrences of {ARGS_KEY}. found {len(norm_data)} {dataset}"
        )
    encoded_data = norm_data[1]
    result = decode_to_json(encoded_data)
    logger.debug(f"Decoded dataset args: {result}")
    dashboard_url = result.get("dashboard_url", None)
    input_norm_fns = get_normalizations(result[INPUT_NORM_DICT_KEY])
    postprocess_fns = get_postprocessors(result[POSTPROCESS_DICT_KEY])
    # Log what actually got built, not the raw dict -- an args block that
    # decodes fine but produces no normalizers is the same silent failure as
    # having no args block at all.
    if not input_norm_fns:
        logger.warning(
            "Dataset args decoded but produced NO input normalizers "
            f"(input_norm={result.get(INPUT_NORM_DICT_KEY)!r}). The model "
            "will see raw voxel values."
        )
    else:
        logger.info(
            f"Serving with input normalizers: "
            f"{[type(fn).__name__ for fn in input_norm_fns]}, postprocessors: "
            f"{[type(fn).__name__ for fn in postprocess_fns]}"
        )
    return dashboard_url, input_norm_fns, postprocess_fns


def serialize_norms_posts_to_json(norms=[], posts=[]):
    norm_fns = {}
    for n in norms:
        elms = n.to_dict()
        elms.pop("name", None)
        norm_fns[n.name()] = elms
    post_fns = {}
    for n in posts:
        elms = n.to_dict()
        elms.pop("name", None)
        post_fns[n.name()] = elms
    return json.dumps({INPUT_NORM_DICT_KEY: norm_fns, POSTPROCESS_DICT_KEY: post_fns})
