// The ranking's filter (cellmap_flow/dashboard/static/js/dashboard/finetune/rank.js
// matchesFilter), which both modes run: spaces, commas, hyphens and
// underscores all separate words, so "mito-aff" finds a model listed as
// mito_aff_unet_setup_16, which it did not.
import { fileURLToPath } from "node:url";
import { expect, it } from "vitest";

const RANK = fileURLToPath(new URL("../../../cellmap_flow/dashboard/static/js/dashboard/finetune/rank.js", import.meta.url));

it("finds a model by its name's words, however they are separated", async () => {
  const { matchesFilter } = (await import(RANK)) as { matchesFilter: (candidate: object, query: string) => boolean };
  const model = { name: "mito_aff_unet_setup_16", task: "huggingface", channels: ["mito_aff_1"], voxel_size_nm: [16, 16, 16], dims: 3 };
  for (const query of ["mito aff", "mito-aff", "mito_aff", "Mito, aff", "aff 16 nm", "setup-16 3d"]) {
    expect(matchesFilter(model, query), query).toBe(true);
  }
  expect(matchesFilter(model, "nuc")).toBe(false);
  expect(matchesFilter(model, "mito 8nm")).toBe(false);
});
