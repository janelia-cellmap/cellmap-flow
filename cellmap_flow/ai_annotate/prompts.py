"""The prompt that asks an image model to paint one structure in an EM plane.

Adapted from ask-to-mask's detailed prompt: what the image is, its
resolution, how the structure looks in EM (when the catalog knows it), and
the instruction to paint it one colour on black. The user may replace the
description and instruction with their own text; the resolution and the
instance-separation sentence are added either way, since the first depends
on the plane actually sent and the second on how the mask is used.
"""

from __future__ import annotations

from cellmap_flow.ai_annotate.errors import AIAnnotateError
from cellmap_flow.ai_annotate.organelles import OrganelleProfile

MAX_PROMPT_CHARS = 4000

# The prompt's fixed sentences, as str.format templates. The browser builds
# the same prompt from these (web/build/export_data.py exports them), so the
# two modes ask the model in the same words.
SENTENCES = {
    "intro": "This is an EM image of cell(s).",
    "appearance": "In EM, {name} appear as: {description}",
    "instruction": (
        "Create a segmentation mask: color all the {name} in {color_name} and make everything else black."
    ),
    # ``resolution`` is formatted ``.3g``: 8 nm says "8", 3.5 nm says "3.5",
    # where ``.0f`` would round it to 4.
    "resolution": "The image resolution is {resolution} nm/px.",
    # Asked of the model rather than fixed afterwards (say, by eroding the
    # mask): the model sees where one instance ends and the next begins,
    # which a fixed erosion only approximates. Accept labels the mask by
    # connected component, which needs this gap to split touching instances
    # instead of merging them.
    "instance_separation": (
        "Where two individual instances are adjacent or touching, leave a thin black "
        "border between them so each instance stays visually separate -- never merge "
        "touching instances into one connected blob."
    ),
    "too_long": "The prompt is longer than {max_chars} characters.",
}


def editable_prompt(profile: OrganelleProfile) -> str:
    """The part of the prompt the user may edit, as the catalog writes it.

    This is what the dashboard prefills the prompt box with. Sending it back
    unedited as ``prompt_override`` gives exactly the prompt no override
    gives, so prefilling never changes what the model is asked.
    """
    parts = [SENTENCES["intro"]]
    if profile.description:
        parts.append(SENTENCES["appearance"].format(name=profile.name, description=profile.description))
    parts.append(SENTENCES["instruction"].format(name=profile.name, color_name=profile.color_name))
    return " ".join(parts)


def build_recolor_prompt(
    profile: OrganelleProfile,
    resolution_nm: float | None = None,
    prompt_override: str | None = None,
) -> str:
    """The recolor prompt for one EM plane.

    The resolution sentence (``resolution_nm`` is the in-plane size of a
    pixel of the image actually sent), then the editable part, then the
    instance-separation sentence. The editable part is ``prompt_override``
    when it has text (a user's dataset-specific wording), else the catalog's
    ``editable_prompt``. An override longer than ``MAX_PROMPT_CHARS`` is
    refused (400); the sentences added around it are short and fixed, so the
    cap is on the part the user writes.
    """
    if prompt_override and prompt_override.strip():
        editable = prompt_override.strip()
        if len(editable) > MAX_PROMPT_CHARS:
            raise AIAnnotateError(
                "refused", SENTENCES["too_long"].format(max_chars=MAX_PROMPT_CHARS), http_status=400
            )
    else:
        editable = editable_prompt(profile)
    parts = []
    if resolution_nm is not None:
        parts.append(SENTENCES["resolution"].format(resolution=f"{resolution_nm:.3g}"))
    parts.append(editable)
    parts.append(SENTENCES["instance_separation"])
    return " ".join(parts)
