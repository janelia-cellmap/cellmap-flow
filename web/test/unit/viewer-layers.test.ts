// The layers (src/viewer/layers.ts) against cellmap_flow/viewer/layers.py and
// bootstrap.py. The transform and dimensions JSON below were printed with the
// pixi python (prediction_source(...) and CoordinateSpace(...).to_json()); the
// shaders through src/viewer/display.ts, itself checked against raw.py in
// viewer-display.test.ts. The chain helpers are the real ones (src/infer/post.ts).
import { describe, expect, it } from "vitest";

import type { Chain, DatasetInfo, ModelEntry, ModelInfo, PaintVolume, ServedModel } from "../../src/contracts";
import { predictionShader, rawShader } from "../../src/viewer/display";
import {
  DEFAULT_LAYOUT, GPU_MEMORY_LIMIT, initialState, PAINT_TOOL_BINDINGS, paintLayer, paintLayerUrl, predictionLayer, RAW_LAYER_NAME, rawLayer,
  rawSource,
} from "../../src/viewer/layers";

const DATASET = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8";

/** A raw volume as the inference module describes it: z, y, x at 8 nm, two levels. */
const info: DatasetInfo = {
  url: DATASET,
  dtype: "uint8",
  channels: 1,
  axes: [{ name: "z", unit: "nm" }, { name: "y", unit: "nm" }, { name: "x", unit: "nm" }],
  levels: [
    { shape: [500, 1000, 2000], voxel: [8, 8, 8], origin: [0, 0, 0] },
    { shape: [250, 500, 1000], voxel: [16, 16, 16], origin: [0, 0, 0] },
  ],
};

function entry(name: string, channels: string[]): ModelEntry {
  const variant = { variant: "fp32" as const, input_size: [36, 36, 36], output_size: [16, 16, 16], file: "model.onnx", file_mb: 12 };
  return {
    id: `cellmap/onnx-web-models/${name}`,
    name,
    repo: "cellmap/onnx-web-models",
    folder: name,
    manifest: {
      model: name, task: "semantic", architecture: "unet", checkpoint: "c", params: 1, channels,
      input_voxel_size_nm: [8, 8, 8], output_voxel_size_nm: [8, 8, 8], output: "logits", input_norm: [],
      io: { input: "x", output: "y" }, variants: [variant],
    },
    variant,
    onnxUrl: `https://huggingface.co/cellmap/onnx-web-models/resolve/main/${name}/model.onnx`,
  };
}

function served(name: string, infoOverrides: Partial<ModelInfo> = {}): ServedModel {
  const channels = infoOverrides.channels ?? ["mito"];
  const modelInfo: ModelInfo = {
    output_channels: channels.length,
    channels,
    write_shape: [16, 16, 16],
    read_shape: [36, 36, 36],
    output_voxel_size: [8, 8, 8],
    input_voxel_size: [8, 8, 8],
    has_channel: true,
    output_axes: ["c", "z", "y", "x"],
    available: true,
    output_class: "unbounded",
    engine: "browser",
    ...infoOverrides,
  };
  return {
    entry: entry(name, channels),
    view: `${name}__CFLOW_ARGS__BLOB__CFLOW_ARGS__`,
    url: `zarr3://https://h/app/virtual/page/${name}__CFLOW_ARGS__BLOB__CFLOW_ARGS__/`,
    info: modelInfo,
    dispose: async () => {},
  };
}

const chain = (...postprocess: Chain["postprocess"]): Chain => ({ input_norm: [], postprocess });
const sigmoid = { name: "SigmoidPostprocessor" };
const threshold = { name: "ThresholdPostprocessor", threshold: "0.5" };

describe("rawSource (raw.neuroglancer_source)", () => {
  it("names the format in front of the location", () => {
    expect(rawSource(DATASET)).toBe(`zarr://${DATASET}`);
    expect(rawSource("s3://bucket/x.zarr/em/s0")).toBe("zarr://s3://bucket/x.zarr/em/s0");
    expect(rawSource("https://h/vol.n5/em")).toBe("n5://https://h/vol.n5/em");
    expect(rawSource("https://h/a.zarr/b.n5/em")).toBe("n5://https://h/a.zarr/b.n5/em"); // the last container wins
    expect(rawSource("https://h/plain/path")).toBe("zarr://https://h/plain/path");
  });
  it("takes a gs:// path with no container as a precomputed volume", () => {
    expect(rawSource("gs://bucket/volume")).toBe("precomputed://gs://bucket/volume");
    expect(rawSource("gs://bucket/x.zarr/em")).toBe("zarr://gs://bucket/x.zarr/em");
  });
});

describe("rawLayer (layers.raw_layer)", () => {
  it("is an image layer named data over the zarr, with the raw shader for the dtype", () => {
    expect(RAW_LAYER_NAME).toBe("data");
    expect(rawLayer(DATASET, info)).toEqual({ type: "image", name: "data", source: `zarr://${DATASET}`, shader: rawShader("uint8") });
    expect(rawLayer(DATASET, { ...info, dtype: "uint16" }).shader).toBe(rawShader("uint16"));
  });
});

describe("predictionLayer (layers.prediction_layer)", () => {
  it("is an image layer in the given colour over the chain's range for a sigmoid chain", () => {
    const model = served("mito");
    expect(predictionLayer(model, chain(sigmoid), "green")).toEqual({
      type: "image",
      name: "mito",
      source: model.url,
      shader: predictionShader("green", [0, 1], "float32"),
    });
  });
  it("is a segmentation layer when the chain ends in labels, shader or not", () => {
    const model = served("mito");
    expect(predictionLayer(model, chain(sigmoid, threshold), "red", { shader: "s" })).toEqual({ type: "segmentation", name: "mito", source: model.url });
  });
  it("transforms a three-channel affinity model drawn at another voxel size than declared", () => {
    const affinities = served("affs", {
      channels: ["z", "y", "x"], output_voxel_size: [16, 16, 16], effective_output_voxel_size: [8, 8, 8], has_channel: true,
    });
    const layer = predictionLayer(affinities, chain(sigmoid), "yellow");
    const dims = { z: [8e-9, "m"], y: [8e-9, "m"], x: [8e-9, "m"], "c^": [1, ""] };
    expect(layer.source).toEqual({ url: affinities.url, transform: { outputDimensions: dims, inputDimensions: dims } });
    expect(layer.type).toBe("image");
    expect(layer.localPosition).toBeUndefined();
  });
});

describe("initialState (bootstrap.new_viewer)", () => {
  it("is the raw's dimensions, the centre, one voxel per pixel, the layers in order, one panel", () => {
    const raw = rawLayer(DATASET, info);
    const model = predictionLayer(served("mito"), chain(sigmoid), "red");
    const state = initialState(info, { data: raw, mito: model });
    expect(state).toEqual({
      gpuMemoryLimit: GPU_MEMORY_LIMIT,
      dimensions: { x: [8e-9, "m"], y: [8e-9, "m"], z: [8e-9, "m"] },
      position: [1000, 500, 250],
      crossSectionScale: 1,
      layers: [{ ...raw, name: "data" }, { ...model, name: "mito" }],
      layout: "xy",
    });
    expect(GPU_MEMORY_LIMIT).toBe(2_000_000_000); // bootstrap.GPU_MEMORY_LIMIT
  });

  it("orders the dimensions x, y, z, so the xy panel is the XY plane (bootstrap.viewer_space)", () => {
    // The "xy" panel shows the first two: in z, y, x order, z across and y down.
    const state = initialState(info, {});
    expect(Object.keys(state.dimensions!)).toEqual(["x", "y", "z"]);
    expect(state.displayDimensions).toBeUndefined();
    expect(DEFAULT_LAYOUT).toBe("xy");
  });
});

describe("the paint layer (overlay.add_annotation_layer)", () => {
  const volume: PaintVolume = {
    id: "vol-0badcafe-20261015-130405",
    type: "annotation_volume",
    model_name: "mito_aff_setup54_16nm",
    model_id: "cellmap/onnx-web-models/setup_54",
    dataset_path: DATASET,
    chunk_size: [60, 60, 60],
    input_size: [148, 148, 148],
    output_voxel_size: [8, 8, 8],
    input_voxel_size: [8, 8, 8],
    dataset_offset_nm: [0, 0, 0],
    dataset_shape_voxels: [1020, 1020, 1020],
    dtype: "uint16",
    input_norm: [],
    postprocess: [],
    created_at: "2026-10-15T13:04:05.000Z",
  };

  it("is a writable segmentation layer, painted through the page's paint route, with the brush and flood fill bound, open on its Draw tab", () => {
    const url = paintLayerUrl(volume.id, "https://janelia-cellmap.github.io/cellmap-flow/app/");
    expect(url).toBe("s3+https://janelia-cellmap.github.io/cellmap-flow/app/paint/vol-0badcafe-20261015-130405/annotation");
    expect(paintLayer(url, volume)).toEqual({
      type: "segmentation",
      name: "annotation_vol-0badcafe-20261015-130405",
      source: { url, subsources: { default: { writingEnabled: true }, bounds: {} } },
      toolBindings: { A: "vox-brush", F: "vox-flood-fill" },
      tab: "Draw",
    });
    // Each layer gets its own bindings object: Neuroglancer may edit it in place.
    expect(paintLayer(url, volume).toolBindings).not.toBe(PAINT_TOOL_BINDINGS);
  });
});
