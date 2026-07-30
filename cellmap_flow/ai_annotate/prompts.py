"""Prompt template for asking Gemini to recolor a target label in an EM crop."""

from __future__ import annotations


def build_recolor_prompt(label_name: str, color_name: str = "bright red") -> str:
    """Build a segmentation-style recolor prompt for a single EM crop.

    Ported from ask-to-mask's OrganelleClass.build_prompt (config.py:32-41),
    trimmed to the one deterministic template this feature needs.
    """
    return (
        "This is an EM image of cell(s). Create a segmentation mask: color all "
        f"the {label_name} in {color_name} and make everything else black."
    )
