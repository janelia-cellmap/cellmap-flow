import logging
from cellmap_flow.utils.web_utils import (
    INPUT_NORM_DICT_KEY,
    POSTPROCESS_DICT_KEY,
)
from cellmap_flow.norm.input_normalize import get_normalizations
from cellmap_flow.post.postprocessors import get_postprocessors
import json

logger = logging.getLogger(__name__)


def get_process_dataset(json_data: dict):
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    logger.error(f"json data: {json_data}")
    input_norm_fns = get_normalizations(json_data[INPUT_NORM_DICT_KEY])
    postprocess_fns = get_postprocessors(json_data[POSTPROCESS_DICT_KEY])
    return input_norm_fns, postprocess_fns


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
