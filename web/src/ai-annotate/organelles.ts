// What the model is told about each structure it can be asked to paint: the
// port of cellmap_flow/ai_annotate/organelles.py. The catalog itself is the
// Python's, exported (generated/python-data.ts): its descriptions are part
// of the prompt, so the browser must send the same words.
//
// A catalog of organelles: the colour to ask the model to paint each one, and
// a description of how it looks in EM to ground the request in what the
// structure actually looks like. Adapted there from ask-to-mask's ORGANELLES
// catalog, keeping only the fields this feature uses. A label name the
// catalog does not know still works (resolveOrganelleProfile), just without
// a description.
//
// The colours are distinct within the catalog but need not be unique across
// it (cells and mitochondria are both red): one request asks for one
// structure.
import { PYTHON } from "../generated/python-data";
import type { OrganelleProfile } from "./types";

const { organelles, aliases, default_color_name: DEFAULT_COLOR_NAME } = PYTHON.ai_annotate;
const DEFAULT_RGB = PYTHON.ai_annotate.default_rgb as [number, number, number];

/** The catalog, in the Python's order (the config route lists it in this order). */
export const ORGANELLES: Readonly<Record<string, OrganelleProfile>> = Object.fromEntries(
  organelles.map((profile) => [profile.key, { ...profile, rgb: profile.rgb as [number, number, number] }]),
);

// Free-text aliases users might type in the label-name field, mapped to a
// catalog key -- covers common synonyms/abbreviations beyond exact name match.
const ALIASES: Readonly<Record<string, string>> = aliases;

/** The catalog profile a free-text label name means, or null
 * (organelles.find_organelle_profile). Matches a key, a name or a known
 * alias, ignoring case, surrounding whitespace, and `-`/`_` versus spaces.
 * null lets the caller fall back to a generic profile. */
export function findOrganelleProfile(labelName: string | null | undefined): OrganelleProfile | null {
  if (!labelName) return null;
  const normalized = labelName.trim().toLowerCase().replaceAll("-", "_");
  if (Object.hasOwn(ORGANELLES, normalized)) return ORGANELLES[normalized];
  const spaced = normalized.replaceAll("_", " ");
  if (Object.hasOwn(ALIASES, spaced)) return ORGANELLES[ALIASES[spaced]];
  for (const profile of Object.values(ORGANELLES)) if (profile.name.toLowerCase() === spaced) return profile;
  return null;
}

/** The catalog profile for `labelName`, or a generic one built from it
 * (organelles.resolve_organelle_profile). A label the catalog does not know
 * still works: the prompt names it as typed (trimmed) and asks for bright
 * red, just without an EM-appearance description. */
export function resolveOrganelleProfile(labelName: string): OrganelleProfile {
  const profile = findOrganelleProfile(labelName);
  if (profile) return profile;
  const name = labelName.trim();
  return { key: name, name, color_name: DEFAULT_COLOR_NAME, rgb: [...DEFAULT_RGB], description: "" };
}
