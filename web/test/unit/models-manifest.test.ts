// parseManifest on the real manifests, the widening of Marwan's short
// fields, the entry naming rule, and the folder filter.
import { describe, expect, it } from "vitest";

import { resolveUrl } from "../../src/models/hf";
import { entryFor, entryName, FALLBACK_FOLDERS, ManifestError, modelFolders, parseManifest, voxelLabel } from "../../src/models/manifest";
import { finetuneModes } from "../../src/finetune/modes";
import { FOLDERS, rawManifest, treeListing } from "./models-helpers";

describe("parseManifest on the repo's manifests", () => {
  it("reads setup_54 (mito affinities, ResUNet3DValid, 16 nm)", () => {
    const m = parseManifest(rawManifest("setup_54"), "setup_54/manifest.json");
    expect(m.model).toBe("setup_54");
    expect(m.task).toBe("mito_affinities");
    expect(m.architecture).toBe("ResUNet3DValid");
    expect(m.checkpoint).toBe("model_checkpoint_260000");
    expect(m.params).toBe(10702083);
    expect(m.channels).toEqual(["mito_aff_z", "mito_aff_y", "mito_aff_x"]);
    expect(m.input_voxel_size_nm).toEqual([16, 16, 16]);
    expect(m.output_voxel_size_nm).toEqual([16, 16, 16]);
    expect(m.output).toBe("raw logits (apply sigmoid / threshold at 0)");
    expect(m.input_norm).toEqual([
      { name: "MinMaxNormalizer", min_value: 0, max_value: 255 },
      { name: "LambdaNormalizer", expression: "2*x-1" },
    ]);
    expect(m.variants).toHaveLength(12);
  });

  it("widens the scalar block sizes to one per spatial axis", () => {
    const m = parseManifest(rawManifest("setup_54"));
    for (const v of m.variants) {
      expect(v.input_size).toHaveLength(3);
      expect(v.output_size).toHaveLength(3);
      // Valid convolutions: the output is 88 voxels smaller on each axis.
      expect(v.output_size.map((n, i) => v.input_size[i]! - n)).toEqual([88, 88, 88]);
    }
    const sizes = new Set(m.variants.map((v) => v.input_size[0]));
    expect([...sizes].sort((a, b) => a! - b!)).toEqual([116, 132, 148]);
  });

  it("reads the tensor names out of the io sentence", () => {
    const m = parseManifest(rawManifest("setup_54"));
    expect(m.io.input).toBe("input");
    expect(m.io.output).toBe("output");
    expect(m.io.description).toBe("input 'input' float32 [1,1,z,y,x]; output 'output' float32 [1,C,z',y',x']");
  });

  it("keeps an io object as given, with defaults for missing names", () => {
    const raw = { ...(rawManifest("setup_54") as Record<string, unknown>), io: { input: "x", note: "n" } };
    expect(parseManifest(raw).io).toEqual({ input: "x", output: "output", note: "n" });
  });

  it("orders the variants smallest block first, fp32 fp16 int8w int8w_fp16 within a block", () => {
    const m = parseManifest(rawManifest("setup_53"));
    expect(m.variants.slice(0, 4).map((v) => [v.input_size[0], v.variant])).toEqual([
      [124, "fp32"],
      [124, "fp16"],
      [124, "int8w"],
      [124, "int8w_fp16"],
    ]);
    expect(m.variants.map((v) => v.input_size[0])).toEqual([124, 124, 124, 124, 140, 140, 140, 140, 172, 172, 172, 172]);
  });

  it("keeps the fidelity numbers and the file size", () => {
    const m = parseManifest(rawManifest("setup_54"));
    const v = m.variants.find((v) => v.variant === "fp16" && v.input_size[0] === 148)!;
    expect(v.file).toBe("setup_54_mito_affinities_16nm_in148_fp16.onnx");
    expect(v.file_mb).toBe(21.55);
    expect(v.max_abs).toBeCloseTo(0.0142, 3);
    expect(v.mean_abs).toBeCloseTo(0.0024, 3);
    expect(v.flip_frac).toBeCloseTo(9.26e-6, 7);
    expect(v.ort_cpu_s).toBe(2.17);
  });

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

  it("keeps the finetune block and the export's provenance, and offers the modes it declares", () => {
    const m = parseManifest(rawManifest("setup_53_2d"));
    expect(m).toMatchObject({ source_repo: "cellmap/mito-aff-unetiso-setup-53-2d", distilled_from_repo: "cellmap/mito-aff-unetiso-setup-53", torch_version: "2.9.1", opset: 18 });
    expect(m.finetune?.head).toMatchObject({ weight: "final_conv.c.weight", bias: "final_conv.c.bias", features: "/final_conv/Gather_output_0" });
    expect(m.finetune?.full).toMatchObject({ library: "tfjs", weight_map: "identity", initializers: "finetune/initializers.json" });
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

  it("takes block sizes already given per axis", () => {
    const raw = rawManifest("setup_54") as { variants: Record<string, unknown>[] };
    const widened = { ...raw, variants: [{ ...raw.variants[0], input_size: [148, 148, 100], output_size: [60, 60, 12] }] };
    const m = parseManifest(widened);
    expect(m.variants[0]!.input_size).toEqual([148, 148, 100]);
    expect(m.variants[0]!.output_size).toEqual([60, 60, 12]);
  });

  it("leaves out a variant of unknown precision and rejects a manifest with none left", () => {
    const raw = rawManifest("setup_54") as { variants: Record<string, unknown>[] };
    const oneOdd = { ...raw, variants: [...raw.variants, { ...raw.variants[0], variant: "bf16" }] };
    expect(parseManifest(oneOdd).variants).toHaveLength(12);
    const allOdd = { ...raw, variants: raw.variants.map((v) => ({ ...v, variant: "bf16" })) };
    expect(() => parseManifest(allOdd, "x")).toThrow(ManifestError);
    expect(() => parseManifest(allOdd, "x")).toThrow(/no variant has a known precision/);
  });

  it("names the field that is missing or wrong", () => {
    const raw = rawManifest("setup_54") as Record<string, unknown>;
    expect(() => parseManifest({ ...raw, channels: [] }, "setup_54/manifest.json")).toThrow("setup_54/manifest.json: channels is not a list of names");
    expect(() => parseManifest({ ...raw, task: undefined })).toThrow(/task is missing/);
    expect(() => parseManifest({ ...raw, input_voxel_size_nm: [16, 16] })).toThrow(/different lengths/);
    expect(() => parseManifest({ ...raw, variants: [] })).toThrow(/variants is missing or empty/);
    expect(() => parseManifest("not an object")).toThrow(/not a JSON object/);
    expect(() => parseManifest(null)).toThrow(ManifestError);
    const badSize = { ...raw, variants: [{ ...(raw.variants as Record<string, unknown>[])[0], input_size: "148" }] };
    expect(() => parseManifest(badSize)).toThrow(/variants\[0\]\.input_size: expected a positive whole number or 3 of them/);
  });
});

describe("entryName", () => {
  it("builds <task>_<model>_<voxel>nm, with affinities shortened", () => {
    expect(entryName(parseManifest(rawManifest("setup_54")))).toBe("mito_aff_setup54_16nm");
    expect(entryName(parseManifest(rawManifest("setup_53")))).toBe("mito_aff_setup53_16nm");
    expect(entryName(parseManifest(rawManifest("setup_56")))).toBe("nuc_setup56_128nm");
    expect(entryName(parseManifest(rawManifest("setup_61")))).toBe("nuc_setup61_256nm");
    expect(entryName(parseManifest(rawManifest("setup_67")))).toBe("er_setup67_8nm");
    // The 2D students keep their tag behind a hyphen: "setup532d" would read as setup 532.
    expect(entryName(parseManifest(rawManifest("setup_53_2d")))).toBe("mito_aff_setup53-2d_16nm");
    expect(entryName(parseManifest(rawManifest("setup_54_2d")))).toBe("mito_aff_setup54-2d_16nm");
  });

  it("is a URL path segment and a Neuroglancer layer name", () => {
    for (const folder of FOLDERS) {
      const name = entryName(parseManifest(rawManifest(folder)));
      expect(name).toMatch(/^[a-z0-9_.-]+$/);
      expect(encodeURIComponent(name)).toBe(name);
      expect(name).not.toContain("__CFLOW_ARGS__");
    }
  });

  it("writes an anisotropic voxel size as z x y x x", () => {
    const m = parseManifest({ ...(rawManifest("setup_54") as Record<string, unknown>), task: "Mito Affinities", input_voxel_size_nm: [16, 8, 8], output_voxel_size_nm: [16, 8, 8] });
    expect(entryName(m)).toBe("mito_aff_setup54_16x8x8nm");
    expect(voxelLabel([5.24, 5.24, 5.24])).toBe("5.24");
    expect(voxelLabel([16, 16, 16])).toBe("16");
  });
});

describe("entryFor", () => {
  it("gives the id, repo, folder and the variant's resolve URL", () => {
    const m = parseManifest(rawManifest("setup_54"));
    const v = m.variants.find((v) => v.variant === "fp32" && v.input_size[0] === 132)!;
    const entry = entryFor("setup_54", m, v);
    expect(entry.id).toBe("cellmap/onnx-web-models/setup_54");
    expect(entry.repo).toBe("cellmap/onnx-web-models");
    expect(entry.folder).toBe("setup_54");
    expect(entry.name).toBe("mito_aff_setup54_16nm");
    expect(entry.variant).toBe(v);
    expect(entry.manifest).toBe(m);
    expect(entry.onnxUrl).toBe("https://huggingface.co/cellmap/onnx-web-models/resolve/main/setup_54/setup_54_mito_affinities_16nm_in132_fp32.onnx");
    expect(entry.onnxUrl).toBe(resolveUrl("setup_54/setup_54_mito_affinities_16nm_in132_fp32.onnx"));
  });
});

describe("modelFolders", () => {
  it("lists the model folders of the real tree, conv3d and the files left out", () => {
    const folders = modelFolders(treeListing() as { type: string; path: string }[]);
    expect(folders).toEqual(FOLDERS);
    expect(folders).toEqual([...FALLBACK_FOLDERS]);
  });
});
