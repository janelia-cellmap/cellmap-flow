// The prompt that asks an image model to paint one structure in an EM plane:
// the port of cellmap_flow/ai_annotate/prompts.py, sentence for sentence (the
// prompt is what the model is asked, so the browser must ask the same).
//
// Adapted there from ask-to-mask's detailed prompt: what the image is, its
// resolution, how the structure looks in EM (when the catalog knows it), and
// the instruction to paint it one colour on black. The user may replace the
// description and instruction with their own text; the resolution and the
// instance-separation sentence are added either way, since the first depends
// on the plane actually sent and the second on how the mask is used.
import { formatG } from "./geometry";
import { AIAnnotateError, MAX_PROMPT_CHARS, type OrganelleProfile } from "./types";

// Asked of the model rather than fixed afterwards (say, by eroding the mask):
// the model sees where one instance ends and the next begins, which a fixed
// erosion only approximates. Accept labels the mask by connected component,
// which needs this gap to split touching instances instead of merging them.
const INSTANCE_SEPARATION_INSTRUCTION =
  "Where two individual instances are adjacent or touching, leave a thin black " +
  "border between them so each instance stays visually separate -- never merge " +
  "touching instances into one connected blob.";

/** `f"The image resolution is {resolution_nm:.3g} nm/px."`: 8 nm says "8",
 * 3.5 nm says "3.5", where a fixed decimal count would round it to 4. */
function resolutionSentence(resolutionNm: number): string {
  return `The image resolution is ${formatG(resolutionNm, 3)} nm/px.`;
}

/** The part of the prompt the user may edit, as the catalog writes it
 * (prompts.editable_prompt). This is what the dashboard prefills the prompt
 * box with. Sending it back unedited as `promptOverride` gives exactly the
 * prompt no override gives, so prefilling never changes what the model is
 * asked. */
export function editablePrompt(profile: OrganelleProfile): string {
  const parts = ["This is an EM image of cell(s)."];
  if (profile.description) parts.push(`In EM, ${profile.name} appear as: ${profile.description}`);
  parts.push(`Create a segmentation mask: color all the ${profile.name} in ${profile.color_name} and make everything else black.`);
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
    if (editable.length > MAX_PROMPT_CHARS) throw new AIAnnotateError("refused", `The prompt is longer than ${MAX_PROMPT_CHARS} characters.`, 400);
  } else {
    editable = editablePrompt(profile);
  }
  const parts: string[] = [];
  if (resolutionNm !== null) parts.push(resolutionSentence(resolutionNm));
  parts.push(editable, INSTANCE_SEPARATION_INSTRUCTION);
  return parts.join(" ");
}
