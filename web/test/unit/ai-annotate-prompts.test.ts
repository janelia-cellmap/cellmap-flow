// The prompt and the catalog (src/ai-annotate/prompts.ts, organelles.ts)
// against cellmap_flow/ai_annotate/prompts.py and organelles.py: the
// sentences and their order, the resolution's spelling, the override and its
// cap, and the catalog's lookups. The parity fixture compares the prompts
// word for word with the Python's.
import { describe, expect, it } from "vitest";

import { findOrganelleProfile, ORGANELLES, resolveOrganelleProfile } from "../../src/ai-annotate/organelles";
import { buildRecolorPrompt, editablePrompt } from "../../src/ai-annotate/prompts";
import { AIAnnotateError, MAX_PROMPT_CHARS } from "../../src/ai-annotate/types";

const SEPARATION = "Where two individual instances are adjacent or touching, leave a thin black border between them so each instance stays visually separate -- never merge touching instances into one connected blob.";

describe("editablePrompt", () => {
  it("names the structure, describes it when the catalog can, and asks for its colour on black", () => {
    const mito = editablePrompt(ORGANELLES.mito);
    expect(mito.startsWith("This is an EM image of cell(s). In EM, mitochondria appear as: large (500-2000 nm), membrane-bound organelles")).toBe(true);
    expect(mito.endsWith("Create a segmentation mask: color all the mitochondria in bright red and make everything else black.")).toBe(true);
    expect(editablePrompt(resolveOrganelleProfile("golgi"))).toBe(
      "This is an EM image of cell(s). Create a segmentation mask: color all the golgi in bright red and make everything else black.",
    );
  });
});

describe("buildRecolorPrompt", () => {
  it("is the resolution sentence, the editable part and the separation instruction", () => {
    expect(buildRecolorPrompt(ORGANELLES.mito, 8)).toBe(`The image resolution is 8 nm/px. ${editablePrompt(ORGANELLES.mito)} ${SEPARATION}`);
    expect(buildRecolorPrompt(ORGANELLES.mito)).toBe(`${editablePrompt(ORGANELLES.mito)} ${SEPARATION}`);
    expect(buildRecolorPrompt(ORGANELLES.mito, null)).toBe(`${editablePrompt(ORGANELLES.mito)} ${SEPARATION}`);
  });

  it("spells the resolution with three significant digits, as :.3g", () => {
    expect(buildRecolorPrompt(ORGANELLES.er, 3.5).startsWith("The image resolution is 3.5 nm/px.")).toBe(true);
    expect(buildRecolorPrompt(ORGANELLES.er, 4).startsWith("The image resolution is 4 nm/px.")).toBe(true);
    expect(buildRecolorPrompt(ORGANELLES.er, 12.3456).startsWith("The image resolution is 12.3 nm/px.")).toBe(true);
    expect(buildRecolorPrompt(ORGANELLES.er, 1024).startsWith("The image resolution is 1.02e+03 nm/px.")).toBe(true);
  });

  it("an override replaces the editable part, trimmed; a blank one does not; the prefilled text changes nothing", () => {
    expect(buildRecolorPrompt(ORGANELLES.mito, 8, "  Paint the dark blobs red.  ")).toBe(`The image resolution is 8 nm/px. Paint the dark blobs red. ${SEPARATION}`);
    expect(buildRecolorPrompt(ORGANELLES.mito, 8, "   ")).toBe(buildRecolorPrompt(ORGANELLES.mito, 8));
    expect(buildRecolorPrompt(ORGANELLES.mito, 8, "")).toBe(buildRecolorPrompt(ORGANELLES.mito, 8));
    expect(buildRecolorPrompt(ORGANELLES.mito, 8, editablePrompt(ORGANELLES.mito))).toBe(buildRecolorPrompt(ORGANELLES.mito, 8));
  });

  it("refuses an override past MAX_PROMPT_CHARS as a 400, and takes one that just fits", () => {
    expect(buildRecolorPrompt(ORGANELLES.mito, 8, "y".repeat(MAX_PROMPT_CHARS))).toContain("y".repeat(MAX_PROMPT_CHARS));
    let caught: unknown;
    try {
      buildRecolorPrompt(ORGANELLES.mito, 8, "x".repeat(MAX_PROMPT_CHARS + 1));
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(AIAnnotateError);
    expect((caught as AIAnnotateError).category).toBe("refused");
    expect((caught as AIAnnotateError).httpStatus).toBe(400);
    expect((caught as AIAnnotateError).userMessage).toBe("The prompt is longer than 4000 characters.");
    // Surrounding whitespace does not count: it is trimmed first.
    expect(() => buildRecolorPrompt(ORGANELLES.mito, 8, `  ${"y".repeat(MAX_PROMPT_CHARS)}  `)).not.toThrow();
  });
});

describe("the catalog", () => {
  it("has eleven structures, each with a key, a plural name, a colour and a description", () => {
    const keys = Object.keys(ORGANELLES);
    expect(keys).toEqual(["mito", "er", "nucleus", "lipid_droplet", "plasma_membrane", "nuclear_envelope", "nuclear_pore", "nucleolus", "heterochromatin", "euchromatin", "cell"]);
    for (const [key, profile] of Object.entries(ORGANELLES)) {
      expect(profile.key).toBe(key);
      expect(profile.name.length).toBeGreaterThan(0);
      expect(profile.color_name.startsWith("bright ")).toBe(true);
      expect(profile.rgb.length).toBe(3);
      for (const c of profile.rgb) expect(c >= 0 && c <= 255 && Number.isInteger(c)).toBe(true);
      expect(profile.description.length).toBeGreaterThan(50);
    }
    expect(ORGANELLES.nuclear_pore.rgb).toEqual([255, 128, 0]);
    expect(ORGANELLES.nucleolus.rgb).toEqual([128, 0, 255]);
    expect(ORGANELLES.cell.rgb).toEqual(ORGANELLES.mito.rgb);  // one request asks for one structure
  });

  it("finds a profile by key, name or alias, ignoring case, whitespace and - versus _", () => {
    expect(findOrganelleProfile("mito")?.key).toBe("mito");
    expect(findOrganelleProfile("Mitochondria")?.key).toBe("mito");
    expect(findOrganelleProfile("mitochondrion")?.key).toBe("mito");
    expect(findOrganelleProfile(" nuclear-pores ")?.key).toBe("nuclear_pore");
    expect(findOrganelleProfile("Cell Membrane")?.key).toBe("plasma_membrane");
    expect(findOrganelleProfile("ENDOPLASMIC RETICULUM")?.key).toBe("er");
    expect(findOrganelleProfile("Lipid Droplet")?.key).toBe("lipid_droplet");
    expect(findOrganelleProfile("cells")?.key).toBe("cell");
    expect(findOrganelleProfile("golgi")).toBeNull();
    expect(findOrganelleProfile("")).toBeNull();
    expect(findOrganelleProfile(null)).toBeNull();
    expect(findOrganelleProfile("mito-chondria")).toBeNull();  // "mito chondria" is no alias
  });

  it("resolves an unknown label to a generic red profile named as typed", () => {
    expect(resolveOrganelleProfile(" golgi apparatus ")).toEqual({ key: "golgi apparatus", name: "golgi apparatus", color_name: "bright red", rgb: [255, 0, 0], description: "" });
    expect(resolveOrganelleProfile("Mito")).toBe(ORGANELLES.mito);
    // Each call's rgb is its own array: a caller may keep it.
    expect(resolveOrganelleProfile("a").rgb).not.toBe(resolveOrganelleProfile("b").rgb);
  });
});
