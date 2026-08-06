"""Known organelle profiles (name, recolor color, EM-appearance description),
ported from ask-to-mask's config.py ORGANELLES catalog, trimmed to the
fields this feature needs: what color to ask Gemini to paint, and a
description to ground it in what the structure actually looks like in EM.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrganelleProfile:
    key: str
    name: str
    color_name: str
    rgb: tuple[int, int, int]
    description: str = ""


_DEFAULT_RGB = (255, 0, 0)
_DEFAULT_COLOR_NAME = "bright red"

ORGANELLES: dict[str, OrganelleProfile] = {
    "mito": OrganelleProfile(
        key="mito",
        name="mitochondria",
        color_name="bright red",
        rgb=(255, 0, 0),
        description=(
            "large (500-2000 nm), membrane-bound organelles, typically ovoid or "
            "elongated (can fuse/branch into tubular networks). Moderate electron "
            "density -- darker than the surrounding cytosol but lighter than "
            "ribosomes. Exact internal texture varies by cell type, prep, and "
            "imaging resolution, so judge primarily by shape, membrane boundary, "
            "and density relative to the surrounding cytosol."
        ),
    ),
    "er": OrganelleProfile(
        key="er",
        name="endoplasmic reticulum",
        color_name="bright green",
        rgb=(0, 255, 0),
        description=(
            "an extensive, interconnected network of flattened cisternae and tubules "
            "(30-60 nm lumen width) that pervades the cytoplasm. Rough ER appears as "
            "parallel membrane pairs studded with dark ribosome dots on the cytoplasmic "
            "face; smooth ER lacks ribosomes and forms more tubular profiles. ER tubules "
            "always connect back to themselves and ultimately connect to the nuclear "
            "envelope. Unlike similar-looking multivesicular bodies or vesicles, ER is "
            "never disconnected—it forms a single continuous network."
        ),
    ),
    "nucleus": OrganelleProfile(
        key="nucleus",
        name="nucleus",
        color_name="bright blue",
        rgb=(0, 0, 255),
        description=(
            "the largest organelle (5-15 µm diameter), roughly spherical or ovoid, "
            "bounded by a double-membrane nuclear envelope perforated with nuclear "
            "pores. The interior contains chromatin: dark, electron-dense heterochromatin "
            "clusters (often along the envelope periphery) and lighter, diffuse "
            "euchromatin. A dense, spherical nucleolus (1-5 µm) is often visible as "
            "the darkest sub-nuclear structure. The nucleoplasm between chromatin "
            "regions appears as a relatively uniform, light gray matrix."
        ),
    ),
    "lipid_droplet": OrganelleProfile(
        key="lipid_droplet",
        name="lipid droplets",
        color_name="bright yellow",
        rgb=(255, 255, 0),
        description=(
            "spherical organelles (100 nm-5 µm) enclosed by a lipid monolayer (not a "
            "bilayer). In heavy-metal stained EM they appear with a shriveled, lumpy "
            "morphology and a generally lighter, more homogeneous interior compared to "
            "the surrounding cytosol. Their boundary shows subtle membrane staining. "
            "Distinguished from lysosomes and endosomes by their lighter, more uniform "
            "interior and absence of internal vesicles or dense granular content."
        ),
    ),
    "plasma_membrane": OrganelleProfile(
        key="plasma_membrane",
        name="plasma membrane",
        color_name="bright cyan",
        rgb=(0, 255, 255),
        description=(
            "a thin (~7-8 nm) dark line delineating the outer boundary of the cell, "
            "separating extracellular space from the cytosol. It appears as a continuous "
            "electron-dense bilayer that follows the entire cell contour, including "
            "microvilli, filopodia, and invaginations. It always forms a closed boundary "
            "around a cell. If a membrane segment appears disconnected from the cell "
            "surface, it is likely part of the endosomal network instead."
        ),
    ),
    "nuclear_envelope": OrganelleProfile(
        key="nuclear_envelope",
        name="nuclear envelope",
        color_name="bright magenta",
        rgb=(255, 0, 255),
        description=(
            "a double-membrane structure (~30-50 nm total thickness) consisting of two "
            "parallel lipid bilayers (inner and outer nuclear membranes) separated by "
            "the perinuclear space. The outer membrane is often studded with ribosomes, "
            "making it continuous with rough ER. It forms a closed boundary around the "
            "nucleus and is perforated by ~120 nm nuclear pore complexes visible as "
            "gaps or electron-dense ring structures. It separates the chromatin-containing "
            "nucleoplasm from the cytosol."
        ),
    ),
    "nuclear_pore": OrganelleProfile(
        key="nuclear_pore",
        name="nuclear pores",
        color_name="bright orange",
        rgb=(255, 128, 0),
        description=(
            "~120 nm diameter protein complexes that perforate the nuclear envelope. "
            "In en-face views they appear as ring-shaped electron-dense structures; in "
            "cross-section they appear as breaks or gaps in envelope connectivity where "
            "the inner and outer nuclear membranes fuse. A central plug or transporter "
            "is sometimes visible. They span both bilayers of the nuclear envelope and "
            "are distributed across the entire nuclear surface."
        ),
    ),
    "nucleolus": OrganelleProfile(
        key="nucleolus",
        name="nucleolus",
        color_name="bright purple",
        rgb=(128, 0, 255),
        description=(
            "a large (1-5 µm), dense, roughly spherical sub-nuclear body that is the "
            "most electron-dense structure within the nucleus. It has a granular and "
            "fibrillar internal texture with distinct sub-compartments: a dense fibrillar "
            "component, a granular component, and sometimes fibrillar centers (lighter "
            "regions). It stains significantly darker than surrounding chromatin and "
            "nucleoplasm. Not bounded by a membrane."
        ),
    ),
    "heterochromatin": OrganelleProfile(
        key="heterochromatin",
        name="heterochromatin",
        color_name="bright spring green",
        rgb=(0, 255, 128),
        description=(
            "dark, electron-dense, compact clusters of tightly packed chromatin within "
            "the nucleus. Typically found along the inner surface of the nuclear envelope "
            "(peripheral heterochromatin) and around the nucleolus (perinucleolar "
            "heterochromatin). Stains significantly darker and is more compact than the "
            "diffuse euchromatin regions. Excludes the nucleolus itself, which has a "
            "distinct granular/fibrillar texture."
        ),
    ),
    "euchromatin": OrganelleProfile(
        key="euchromatin",
        name="euchromatin",
        color_name="bright rose",
        rgb=(255, 0, 128),
        description=(
            "light, diffuse, loosely packed chromatin that fills much of the nuclear "
            "interior between heterochromatin clusters. Appears as a lighter gray, more "
            "uniform matrix compared to the dark heterochromatin. Represents "
            "transcriptionally active chromatin regions. Located outside of the nucleolus "
            "and distinct from the dense nucleolar sub-structure. Excludes chromatin "
            "directly associated with the nucleolus."
        ),
    ),
    "cell": OrganelleProfile(
        key="cell",
        name="cells",
        color_name="bright red",
        rgb=(255, 0, 0),
        description=(
            "whole cells bounded by a plasma membrane. Each cell contains cytoplasm "
            "filled with various organelles (mitochondria, ER, nucleus, etc.) and a "
            "relatively uniform cytosolic matrix. Cell boundaries are defined by the "
            "plasma membrane—a thin dark line separating one cell from another or from "
            "extracellular space. Extracellular regions may appear lighter or contain "
            "extracellular matrix material."
        ),
    ),
}

# Free-text aliases users might type in the label-name field, mapped to a
# catalog key -- covers common synonyms/abbreviations beyond exact name match.
_ALIASES: dict[str, str] = {
    "mitochondria": "mito",
    "mitochondrion": "mito",
    "endoplasmic reticulum": "er",
    "lipid droplet": "lipid_droplet",
    "lipid droplets": "lipid_droplet",
    "plasma membrane": "plasma_membrane",
    "cell membrane": "plasma_membrane",
    "nuclear envelope": "nuclear_envelope",
    "nuclear pore": "nuclear_pore",
    "nuclear pores": "nuclear_pore",
    "cells": "cell",
}


def find_organelle_profile(label_name: str) -> OrganelleProfile | None:
    """Best-effort match of a free-text label name to a known organelle
    profile (exact key, exact name, or a known alias) -- normalizes case and
    surrounding whitespace/underscores. Returns None if nothing matches, so
    the caller can fall back to a generic profile.
    """
    if not label_name:
        return None
    normalized = label_name.strip().lower().replace("-", "_")
    if normalized in ORGANELLES:
        return ORGANELLES[normalized]
    spaced = normalized.replace("_", " ")
    if spaced in _ALIASES:
        return ORGANELLES[_ALIASES[spaced]]
    for profile in ORGANELLES.values():
        if profile.name.lower() == spaced:
            return profile
    return None


def resolve_organelle_profile(label_name: str) -> OrganelleProfile:
    """Match label_name to a known profile, or build a generic one from the
    raw label text so unrecognized labels still work (just without a
    detailed EM-appearance description).
    """
    profile = find_organelle_profile(label_name)
    if profile is not None:
        return profile
    return OrganelleProfile(
        key=label_name,
        name=label_name,
        color_name=_DEFAULT_COLOR_NAME,
        rgb=_DEFAULT_RGB,
        description="",
    )
