// The prompt that asks an image model to paint one structure in an EM plane:
// the port of cellmap_flow/ai_annotate/prompts.py, built from its own
// sentences (prompts.SENTENCES, exported in generated/python-data.ts): the
// prompt is what the model is asked, so the browser asks in the same words.
//
// Adapted there from ask-to-mask's detailed prompt: what the image is, its
// resolution, how the structure looks in EM (when the catalog knows it), and
// the instruction to paint it one colour on black. The user may replace the
// description and instruction with their own text; the resolution and the
// instance-separation sentence are added either way, since the first depends
// on the plane actually sent and the second on how the mask is used.
import { PYTHON, pyFormat } from "../generated/python-data";
import { formatG } from "./geometry";
import { AIAnnotateError, MAX_PROMPT_CHARS, type OrganelleProfile } from "./types";

const SENTENCES = PYTHON.ai_annotate.sentences;

/** The part of the prompt the user may edit, as the catalog writes it
 * (prompts.editable_prompt). This is what the dashboard prefills the prompt
 * box with. Sending it back unedited as `promptOverride` gives exactly the
 * prompt no override gives, so prefilling never changes what the model is
 * asked. */
export function editablePrompt(profile: OrganelleProfile): string {
  const parts = [SENTENCES.intro];
  if (profile.description) parts.push(pyFormat(SENTENCES.appearance, { name: profile.name, description: profile.description }));
  parts.push(pyFormat(SENTENCES.instruction, { name: profile.name, color_name: profile.color_name }));
  return parts.join(" ");
}

/** The recolor prompt for one EM plane (prompts.build_recolor_prompt).
 *
 * The resolution sentence (`resolutionNm` is the in-plane size of a pixel
 * of the image actually sent; none when null), then the editable part, then
 * the instance-separation sentence. The editable part is `promptOverride`
 * when it has text (a user's dataset-specific wording, trimmed), else the
 * catalog's editablePrompt. An override longer than MAX_PROMPT_CHARS is
 * refused (AIAnnotateError "refused", 400); the sentences added around it
 * are short and fixed, so the cap is on the part the user writes. */
export function buildRecolorPrompt(profile: OrganelleProfile, resolutionNm: number | null = null, promptOverride: string | null = null): string {
  let editable: string;
  if (promptOverride && promptOverride.trim()) {
    editable = promptOverride.trim();
    if (editable.length > MAX_PROMPT_CHARS) throw new AIAnnotateError("refused", pyFormat(SENTENCES.too_long, { max_chars: String(MAX_PROMPT_CHARS) }), 400);
  } else {
    editable = editablePrompt(profile);
  }
  const parts: string[] = [];
  // `.3g`, as the Python formats it: 8 nm says "8", 3.5 nm says "3.5".
  if (resolutionNm !== null) parts.push(pyFormat(SENTENCES.resolution, { resolution: formatG(resolutionNm, 3) }));
  parts.push(editable, SENTENCES.instance_separation);
  return parts.join(" ");
}
