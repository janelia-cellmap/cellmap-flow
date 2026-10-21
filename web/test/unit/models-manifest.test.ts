// parseManifest on the real manifests, the finetune modes a manifest offers
// for the file served, the refusal of a manifest that is wrong, and the
// entry names as the URLs and the viewer take them.
import { describe, expect, it } from "vitest";

import { entryFor, entryName, ManifestError, parseManifest } from "../../src/models/manifest";
import { finetuneModes } from "../../src/finetune/modes";
import { FOLDERS, rawManifest } from "./models-helpers";

describe("parseManifest on the repo's manifests", () => {
  it("parses every folder's manifest", () => {
    for (const folder of FOLDERS) {
      const m = parseManifest(rawManifest(folder), folder);
      expect(m.model).toBe(folder);
      // Twelve files per 3D folder (three blocks × four precisions); six per
      // 2D student (three blocks × fp32, fp16: no int8 export).
      expect(m.variants).toHaveLength(folder.endsWith("_2d") ? 6 : 12);
      expect(m.channels.length).toBeGreaterThan(0);
    }
  });

  it("offers the finetune modes it declares, as the file served allows", () => {
    // The modes follow the file served: a 3D model's int8w file has no
    // writable weights; a manifest without the block keeps the head.
    const modes = (folder: string, variant: string, raw = rawManifest(folder)) => {
      const manifest = parseManifest(raw);
      return finetuneModes(entryFor(folder, manifest, manifest.variants.find((v) => v.variant === variant)!));
    };
    expect(modes("setup_53_2d", "fp16")).toEqual(["head", "full"]);
    expect(modes("setup_54", "fp32")).toEqual(["head"]);
    expect(modes("setup_54", "int8w")).toEqual([]);
    const before = { ...(rawManifest("setup_54") as Record<string, unknown>) };
    delete before.finetune;
    expect(modes("setup_54", "int8w", before)).toEqual(["head"]);
    // A 3D model's manifest declaring the whole network (NDHWC, since
    // 2026-10-17): offered with the files that map its PyTorch-named params
    // onto the initializers, and not without them.
    const raw54 = rawManifest("setup_54") as { finetune: Record<string, unknown> };
    const full = { library: "tfjs", files: ["finetune/train_graph.json"], weight_map: "finetune/weight_map.json", torch_params: "finetune/torch_params.json", initializers: "finetune/initializers.json", write_back: [116, 132, 148].map((n) => `setup_54_mito_affinities_16nm_in${n}_fp32.onnx`) };
    const with3d = (spec: Record<string, unknown>) => ({ ...raw54, finetune: { ...raw54.finetune, full: spec } });
    expect(modes("setup_54", "fp32", with3d(full))).toEqual(["head", "full"]);
    expect(modes("setup_54", "fp32", with3d({ ...full, torch_params: undefined }))).toEqual(["head"]);
    expect(modes("setup_54", "fp32", with3d({ ...full, weight_map: "identity" }))).toEqual(["head"]);
  });

  it("names the field that is missing or wrong", () => {
    const raw = rawManifest("setup_54") as Record<string, unknown>;
    expect(() => parseManifest({ ...raw, channels: [] }, "setup_54/manifest.json")).toThrow(/channels/);
    expect(() => parseManifest({ ...raw, task: undefined })).toThrow(/task/);
    expect(() => parseManifest({ ...raw, input_voxel_size_nm: [16, 16] })).toThrow(ManifestError);
    expect(() => parseManifest({ ...raw, variants: [] })).toThrow(/variants/);
    expect(() => parseManifest("not an object")).toThrow(ManifestError);
    expect(() => parseManifest(null)).toThrow(ManifestError);
    const badSize = { ...raw, variants: [{ ...(raw.variants as Record<string, unknown>[])[0], input_size: "148" }] };
    expect(() => parseManifest(badSize)).toThrow(/input_size/);
  });
});

describe("entryName", () => {
  it("is a URL path segment and a Neuroglancer layer name", () => {
    for (const folder of FOLDERS) {
      const name = entryName(parseManifest(rawManifest(folder)));
      expect(name).toMatch(/^[a-z0-9_.-]+$/);
      expect(encodeURIComponent(name)).toBe(name);
      expect(name).not.toContain("__CFLOW_ARGS__");
    }
  });
});
