// A link that opens a volume and runs models (src/link.ts): its query
// parsed, taken off the URL, and its models resolved through the catalog.
import { describe, expect, it } from "vitest";

import type { ModelEntry } from "../../src/contracts";
import { parseLink, resolveLinkedModels, withoutLink } from "../../src/link";

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
