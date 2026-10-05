"""The organelle catalog and the recolor prompt built from it."""

import pytest

from cellmap_flow.ai_annotate.errors import AIAnnotateError
from cellmap_flow.ai_annotate.organelles import ORGANELLES, find_organelle_profile, resolve_organelle_profile
from cellmap_flow.ai_annotate.prompts import MAX_PROMPT_CHARS, build_recolor_prompt, editable_prompt


@pytest.mark.parametrize(
    "label, key",
    [("mito", "mito"), ("Mitochondria", "mito"), (" lipid-droplet ", "lipid_droplet"), ("Lipid Droplets", "lipid_droplet"),
     ("endoplasmic reticulum", "er"), ("cell membrane", "plasma_membrane"), ("nucleolus", "nucleolus")],
)
def test_label_names_find_their_profile(label, key):
    assert find_organelle_profile(label) is ORGANELLES[key]


def test_an_unknown_label_gets_a_generic_red_profile():
    assert find_organelle_profile("centrioles") is None
    assert find_organelle_profile("") is None
    profile = resolve_organelle_profile("  centrioles ")
    assert (profile.name, profile.rgb, profile.description) == ("centrioles", (255, 0, 0), "")


def test_the_prompt_states_the_resolution_sent_and_the_colour():
    prompt = build_recolor_prompt(ORGANELLES["mito"], resolution_nm=8.0)
    assert "8 nm/px" in prompt
    assert "mitochondria in bright red" in prompt
    assert "In EM, mitochondria appear as:" in prompt
    assert "leave a thin black border" in prompt
    assert "3.5 nm/px" in build_recolor_prompt(ORGANELLES["mito"], resolution_nm=3.5)
    assert "nm/px" not in build_recolor_prompt(ORGANELLES["mito"])


def test_a_generic_profile_has_no_description_sentence():
    prompt = build_recolor_prompt(resolve_organelle_profile("centrioles"))
    assert "In EM" not in prompt and "color all the centrioles in bright red" in prompt


def test_an_override_replaces_the_middle_but_keeps_resolution_and_separation():
    prompt = build_recolor_prompt(ORGANELLES["er"], resolution_nm=16, prompt_override="  Paint the ER green.  ")
    assert prompt.startswith("The image resolution is 16 nm/px. Paint the ER green. ")
    assert "leave a thin black border" in prompt
    assert "In EM" not in prompt


def test_a_blank_override_is_no_override():
    assert build_recolor_prompt(ORGANELLES["er"], 8, "   ") == build_recolor_prompt(ORGANELLES["er"], 8)


def test_sending_the_prefilled_text_back_unedited_changes_nothing():
    for profile in ORGANELLES.values():
        assert build_recolor_prompt(profile, 8, editable_prompt(profile)) == build_recolor_prompt(profile, 8)


def test_an_override_over_the_cap_is_refused():
    build_recolor_prompt(ORGANELLES["mito"], 8, "x" * MAX_PROMPT_CHARS)
    with pytest.raises(AIAnnotateError) as caught:
        build_recolor_prompt(ORGANELLES["mito"], 8, "x" * (MAX_PROMPT_CHARS + 1))
    assert caught.value.http_status == 400
