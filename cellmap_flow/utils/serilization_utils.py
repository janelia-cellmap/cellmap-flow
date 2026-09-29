import json
import logging

from cellmap_flow.pipeline_spec import PipelineSpec, split_dataset_url
from cellmap_flow.utils.web_utils import (
    ARGS_KEY,
    INPUT_NORM_DICT_KEY,
    decode_to_json,
)

logger = logging.getLogger(__name__)


def get_process_dataset(json_data: dict):
    """``(input_norms, postprocess)`` built from a ``json_data`` dict or its JSON.

    Either chain may be in the list or the legacy dict form; both keys are
    required (the blockwise precheck reports a json_data without them).
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    logger.info(f"json data: {json_data}")
    return PipelineSpec.from_json_data(json_data, strict=True).build()


def get_process_dataset_url(dataset: str):
    blob = split_dataset_url(dataset)
    if blob is None:
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
    # Decoded here rather than through PipelineSpec.from_url_blob so the
    # messages below can show the chain exactly as the URL spelled it.
    result = decode_to_json(blob)
    logger.debug(f"Decoded dataset args: {result}")
    dashboard_url = result.get("dashboard_url", None)
    input_norm_fns, postprocess_fns = PipelineSpec.from_json_data(
        result, strict=True
    ).build()
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


def serialize_norms_posts_to_json(norms=(), posts=()):
    """JSON with both chains in the ordered ``[{name, **params}]`` form."""
    return json.dumps(PipelineSpec.from_steps(norms, posts).to_json_data())
