// A link that opens a volume and runs models (src/link.ts): its query
// parsed, taken off the URL, and its models resolved through the catalog.
import { describe, expect, it } from "vitest";

import { ARGS_KEY, type ModelEntry } from "../../src/contracts";
import { chainFromHash, parseLink, resolveLinkedModels, withoutLink } from "../../src/link";
import { encodeArgs } from "../../src/shared/args";
import { encodeState } from "../../src/viewer/share";

const DATASET = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8";

describe("parseLink", () => {
  it("reads the dataset and the models, repeated or comma-separated, each once", () => {
    expect(parseLink(`?dataset=${DATASET}&model=setup_53`)).toEqual({ datasetUrl: DATASET, models: ["setup_53"] });
    expect(parseLink("?model=setup_53&model=setup_54,setup_53&dataset=")).toEqual({ datasetUrl: null, models: ["setup_53", "setup_54"] });
    expect(parseLink("")).toEqual({ datasetUrl: null, models: [] });
    expect(parseLink("?other=1")).toEqual({ datasetUrl: null, models: [] });
  });

  it("takes the dataset URL encoded or plain", () => {
    expect(parseLink(`?dataset=${encodeURIComponent(DATASET)}`).datasetUrl).toBe(DATASET);
    expect(parseLink(`?dataset=${DATASET}`).datasetUrl).toBe(DATASET);
  });
});

describe("withoutLink", () => {
  it("drops the link's parameters and keeps the rest of the URL", () => {
    expect(withoutLink(`https://h/app/?dataset=${encodeURIComponent(DATASET)}&model=setup_53`)).toBe("https://h/app/");
    expect(withoutLink("https://h/app/?dataset=x&keep=1&model=a#!state")).toBe("https://h/app/?keep=1#!state");
    expect(withoutLink("https://h/app/")).toBe("https://h/app/");
  });
});

describe("chainFromHash", () => {
  const chain = {
    input_norm: [{ name: "MinMaxNormalizer", min_value: "0", max_value: "255" }],
    postprocess: [{ name: "ThresholdPostprocessor", threshold: "0.5" }],
  };
  const raw = { type: "image", source: "zarr://https://x/data.zarr/", name: "data" };

  it("reads the chain a model layer's view was drawn through", () => {
    const view = `mito_aff_setup53_16nm${ARGS_KEY}${encodeArgs(chain, { digest: "abc" })}${ARGS_KEY}`;
    const model = { type: "image", source: `zarr3://https://h/app/virtual/a30d4ff4/${view}/`, name: "mito_aff_setup53_16nm" };
    expect(chainFromHash(encodeState({ layers: [raw, model] }))).toEqual(chain);
  });

  it("is null without such a layer, or without a state", () => {
    expect(chainFromHash(encodeState({ layers: [raw] }))).toBeNull();
    expect(chainFromHash("")).toBeNull();
    expect(chainFromHash("#!not-a-state")).toBeNull();
  });
});

describe("resolveLinkedModels", () => {
  const setup53 = { id: "cellmap/onnx-web-models/setup_53", name: "setup_53" } as unknown as ModelEntry;

  it("resolves what the catalog knows, logs what it does not, and keeps each model once", async () => {
    const lines: string[] = [];
    const catalog = { resolve: async (ref: string) => (ref.endsWith("setup_53") ? setup53 : null) };
    const entries = await resolveLinkedModels(catalog, ["setup_53", "nope", "cellmap/onnx-web-models/setup_53"], (l) => lines.push(l));
    expect(entries).toEqual([setup53]);
    expect(lines).toEqual([expect.stringContaining("nope")]);
  });

  it("logs a lookup that fails and goes on", async () => {
    const lines: string[] = [];
    const catalog = { resolve: async (ref: string) => { if (ref === "bad") throw new Error("offline"); return setup53; } };
    const entries = await resolveLinkedModels(catalog, ["bad", "setup_53"], (l) => lines.push(l));
    expect(entries).toEqual([setup53]);
    expect(lines).toEqual([expect.stringContaining("offline"), expect.stringContaining("bad")]);
  });
});
