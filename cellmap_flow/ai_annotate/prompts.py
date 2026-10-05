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

# Asked of the model rather than fixed afterwards (say, by eroding the mask):
# the model sees where one instance ends and the next begins, which a fixed
# erosion only approximates. Accept labels the mask by connected component,
# which needs this gap to split touching instances instead of merging them.
_INSTANCE_SEPARATION_INSTRUCTION = (
    "Where two individual instances are adjacent or touching, leave a thin black "
    "border between them so each instance stays visually separate -- never merge "
    "touching instances into one connected blob."
)


def _resolution_sentence(resolution_nm):
    # ``g``: 8 nm says "8", 3.5 nm says "3.5", where ``.0f`` would round it to 4.
    return f"The image resolution is {resolution_nm:.3g} nm/px."


def editable_prompt(profile: OrganelleProfile) -> str:
    """The part of the prompt the user may edit, as the catalog writes it.

    This is what the dashboard prefills the prompt box with. Sending it back
    unedited as ``prompt_override`` gives exactly the prompt no override
    gives, so prefilling never changes what the model is asked.
    """
    parts = ["This is an EM image of cell(s)."]
    if profile.description:
        parts.append(f"In EM, {profile.name} appear as: {profile.description}")
    parts.append(
        f"Create a segmentation mask: color all the {profile.name} in "
        f"{profile.color_name} and make everything else black."
    )
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
                "refused", f"The prompt is longer than {MAX_PROMPT_CHARS} characters.", http_status=400
            )
    else:
        editable = editable_prompt(profile)
    parts = []
    if resolution_nm is not None:
        parts.append(_resolution_sentence(resolution_nm))
    parts.append(editable)
    parts.append(_INSTANCE_SEPARATION_INSTRUCTION)
    return " ".join(parts)
