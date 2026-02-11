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
        return None, [], []  # No normalization or postprocessing
    norm_data = dataset.split(ARGS_KEY)
    if len(norm_data) != 3:
        raise ValueError(
            f"Invalid dataset format. Expected two occurrences of {ARGS_KEY}. found {len(norm_data)} {dataset}"
        )
    encoded_data = norm_data[1]
    result = decode_to_json(encoded_data)
    logger.error(f"Decoded data: {result}")
    dashboard_url = result.get("dashboard_url", None)
    input_norm_fns = get_normalizations(result[INPUT_NORM_DICT_KEY])
    postprocess_fns = get_postprocessors(result[POSTPROCESS_DICT_KEY])
    logger.error(f"Normalized data: {result}")
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
