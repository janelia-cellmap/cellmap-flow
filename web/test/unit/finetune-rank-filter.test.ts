// The search of the ranking's sections (cellmap_flow/dashboard/static/js/
// dashboard/finetune/rank.js candidateText, what a row is searched by), which
// both modes run: spaces, commas, hyphens and underscores all separate words,
// so "mito-aff" finds a model listed as mito_aff_unet_setup_16, which it did
// not.
import { fileURLToPath } from "node:url";
import { expect, it } from "vitest";

const RANK = fileURLToPath(new URL("../../../cellmap_flow/dashboard/static/js/dashboard/finetune/rank.js", import.meta.url));
// The Models tab's Hugging Face list searches by the same rule (lib/search.js):
// what was typed was matched as one string, so "mito aff" found nothing.
const SEARCH = fileURLToPath(new URL("../../../cellmap_flow/dashboard/static/js/lib/search.js", import.meta.url));

it("finds a model by its name's words, however they are separated", async () => {
  const { candidateText } = (await import(RANK)) as { candidateText: (candidate: object) => string };
  const { matchesWords } = (await import(SEARCH)) as { matchesWords: (text: string, query: string) => boolean };
  const text = candidateText({ name: "mito_aff_unet_setup_16", task: "huggingface", channels: ["mito_aff_1"], voxel_size_nm: [16, 16, 16], dims: 3 });
  for (const query of ["mito aff", "mito-aff", "mito_aff", "Mito, aff", "aff 16 nm", "setup-16 3d"]) {
    expect(matchesWords(text, query), query).toBe(true);
  }
  expect(matchesWords(text, "nuc")).toBe(false);
  expect(matchesWords(text, "mito 8nm")).toBe(false);
});

it("finds a Hugging Face repo by its words and by 2d, as the ranking does", async () => {
  const { dimsWord, matchesWords, searchText } = (await import(SEARCH)) as {
    dimsWord: (shape: unknown) => string | null; matchesWords: (text: string, query: string) => boolean; searchText: (...parts: unknown[]) => string;
  };
  const text = searchText("cellmap/mito-aff-unetiso-setup-53-2d", "16nm", dimsWord([1, 556, 556]));
  for (const query of ["mito aff", "setup 53 2d", "16 nm", "MITO-AFF"]) expect(matchesWords(text, query), query).toBe(true);
  expect(matchesWords(searchText("cellmap/nuc-resunet-setup-56", dimsWord([148, 148, 148])), "2d")).toBe(false);
});
