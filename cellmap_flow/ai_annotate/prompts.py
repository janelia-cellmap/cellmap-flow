"""Prompt template for asking Gemini to recolor a target organelle/structure
in an EM crop, ported from ask-to-mask's OrganelleClass.build_prompt (with
detailed=True): EM preamble + optional resolution + optional EM-appearance
description (when the label matches a known organelle profile) + the
recolor instruction itself.
"""

from __future__ import annotations

from cellmap_flow.ai_annotate.organelles import OrganelleProfile

# Asked of Gemini directly rather than fixed up after the fact (e.g. via
# post-hoc mask erosion): Gemini can see individual instance boundaries and
# draw the gap correctly, whereas a fixed erosion is a blunt approximation.
# Downstream training (crop_loader.py's connected_components=True import
# path) runs scipy.ndimage.label on the foreground mask, which needs this
# gap to split touching instances instead of merging them into one.
_INSTANCE_SEPARATION_INSTRUCTION = (
    "Where two individual instances are adjacent or touching, leave a thin black "
    "border between them so each instance stays visually separate -- never merge "
    "touching instances into one connected blob."
)


def build_recolor_prompt(
    profile: OrganelleProfile,
    resolution_nm: float | None = None,
    prompt_override: str | None = None,
) -> str:
    """Build a segmentation-style recolor prompt for a single EM crop.

    ``prompt_override``, when set, replaces the auto-built description/
    instruction text wholesale (e.g. a user's dataset-specific edit made in
    the UI before ever calling Gemini) -- resolution is still prepended
    dynamically since it depends on the crop actually fetched at request
    time, not on the dataset-agnostic template. The instance-separation
    instruction is always appended, override or not.
    """
    if prompt_override and prompt_override.strip():
        parts = []
        if resolution_nm is not None:
            parts.append(f"The image resolution is {resolution_nm:.0f}nm/px.")
        parts.append(prompt_override.strip())
        parts.append(_INSTANCE_SEPARATION_INSTRUCTION)
        return " ".join(parts)

    parts = ["This is an EM image of cell(s)."]
    if resolution_nm is not None:
        parts.append(f"The image resolution is {resolution_nm:.0f}nm/px.")
    if profile.description:
        parts.append(f"In EM, {profile.name} appear as: {profile.description}")
    parts.append(
        f"Create a segmentation mask: color all the {profile.name} in "
        f"{profile.color_name} and make everything else black."
    )
    parts.append(_INSTANCE_SEPARATION_INSTRUCTION)
    return " ".join(parts)
