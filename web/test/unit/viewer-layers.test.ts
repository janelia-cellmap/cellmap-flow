// The layers (src/viewer/layers.ts) against cellmap_flow/viewer/layers.py and
// bootstrap.py. The transform and dimensions JSON below were printed with the
// pixi python (prediction_source(...) and CoordinateSpace(...).to_json()); the
// shaders through src/viewer/display.ts, itself checked against raw.py in
// viewer-display.test.ts. The chain helpers are the real ones (src/infer/post.ts).
import { describe, expect, it } from "vitest";

import type { Chain, DatasetInfo, ModelEntry, ModelInfo, ServedModel } from "../../src/contracts";
import { predictionShader, rawShader } from "../../src/viewer/display";
import {
  centrePosition, DEFAULT_LAYOUT, dimensionsOf, displayChannel, GPU_MEMORY_LIMIT, initialState, layerList, PREDICTION_COLORS,
  predictionLayer, predictionSource, predictionVoxelOverride, RAW_LAYER_NAME, rawLayer, rawSource, stateWithoutInfo,
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
const defaults = { name: "DefaultPostprocessor", clip_min: "-1", clip_max: "1", bias: "1", multiplier: "127.5" };

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
  it("keeps a source that already names its format", () => {
    expect(rawSource("zarr3://https://h/x/")).toBe("zarr3://https://h/x/");
    expect(rawSource("precomputed://https://h/vol")).toBe("precomputed://https://h/vol");
  });
});

describe("rawLayer (layers.raw_layer)", () => {
  it("is an image layer named data over the zarr, with the raw shader for the dtype", () => {
    expect(RAW_LAYER_NAME).toBe("data");
    expect(rawLayer(DATASET, info)).toEqual({ type: "image", name: "data", source: `zarr://${DATASET}`, shader: rawShader("uint8") });
    expect(rawLayer(DATASET, { ...info, dtype: "uint16" }).shader).toBe(rawShader("uint16"));
  });
  it("puts the user's shader and controls back", () => {
    const keep = { shader: "void main(){emitGrayscale(normalized());}", shaderControls: { normalized: { range: [20, 200] } } };
    expect(rawLayer(DATASET, info, keep)).toEqual({ type: "image", name: "data", source: `zarr://${DATASET}`, ...keep });
    expect(rawLayer(DATASET, info, { shaderControls: keep.shaderControls })).toMatchObject({ shader: rawShader("uint8"), shaderControls: keep.shaderControls });
  });
  it("ignores the Python API's \"None\" shader and empty controls", () => {
    expect(rawLayer(DATASET, info, { shader: "None", shaderControls: {} })).toEqual(rawLayer(DATASET, info));
  });
});

describe("predictionVoxelOverride and predictionSource (layers.prediction_source)", () => {
  it("has no override while the engine draws the output at its declared size", () => {
    expect(predictionVoxelOverride(served("m").info)).toBeNull();
    expect(predictionVoxelOverride(served("m", { effective_output_voxel_size: [8, 8, 8] }).info)).toBeNull();
    expect(predictionSource("zarr3://u/", null, true)).toBe("zarr3://u/");
  });
  it("overrides with the effective size when it differs", () => {
    expect(predictionVoxelOverride(served("m", { output_voxel_size: [16, 16, 16], effective_output_voxel_size: [12, 12, 12] }).info)).toEqual([12, 12, 12]);
  });
  it("declares the array at the override scales, the channel axis kept as c^", () => {
    // prediction_source("http://h", "m", "BLOB", (8.0, 8.0, 8.0), has_channel=True, engine="chunkmirage")
    const dims = { z: [8e-9, "m"], y: [8e-9, "m"], x: [8e-9, "m"], "c^": [1, ""] };
    expect(predictionSource("zarr3://u/", [8, 8, 8], true)).toEqual({ url: "zarr3://u/", transform: { outputDimensions: dims, inputDimensions: dims } });
    // has_channel=False
    const spatial = { z: [8e-9, "m"], y: [8e-9, "m"], x: [8e-9, "m"] };
    expect(predictionSource("zarr3://u/", [8, 8, 8], false)).toEqual({ url: "zarr3://u/", transform: { outputDimensions: spatial, inputDimensions: spatial } });
  });
});

describe("displayChannel (layers.display_channel)", () => {
  const cellpose = served("cyto", { channels: ["flow_y", "flow_x", "cell"], display_channel: 2 });
  it("is the engine's display channel while the chain keeps the channels", () => {
    expect(displayChannel(cellpose.info, [])).toBe(2);
    expect(displayChannel(cellpose.info, [sigmoid])).toBe(2);
  });
  it("is null when the chain picks channels, without a channel axis, or when none is named", () => {
    expect(displayChannel(cellpose.info, [{ name: "ChannelSelection", channels: "2" }])).toBeNull();
    expect(displayChannel({ ...cellpose.info, has_channel: false }, [])).toBeNull();
    expect(displayChannel(served("m").info, [])).toBeNull();
    expect(displayChannel({ ...cellpose.info, display_channel: null }, [])).toBeNull();
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
  it("shades a DefaultPostprocessor after a sigmoid over 127.5..255 as uint8", () => {
    const layer = predictionLayer(served("mito"), chain(sigmoid, defaults), "red");
    expect(layer.shader).toBe(predictionShader("red", [127.5, 255], "uint8"));
    expect(layer.shader).toContain("range=[127, 255], window=[63, 255]");
  });
  it("falls back to [0, 1] for raw logits, whose range is undetermined", () => {
    expect(predictionLayer(served("mito"), chain(), "blue").shader).toBe(predictionShader("blue", null, "float32"));
    expect(predictionLayer(served("mito", { output_class: "signed_unit" }), chain(), "blue").shader).toBe(predictionShader("blue", [-1, 1], "float32"));
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
  it("leaves a three-channel model at its declared size as a plain source", () => {
    const affinities = served("affs", { channels: ["z", "y", "x"] });
    expect(predictionLayer(affinities, chain(sigmoid), "yellow").source).toBe(affinities.url);
  });
  it("opens on the engine's display channel, at its centre", () => {
    const cellpose = served("cyto", { channels: ["flow_y", "flow_x", "cell"], display_channel: 2 });
    expect(predictionLayer(cellpose, chain(), "red").localPosition).toEqual([2.5]);
    expect(predictionLayer(cellpose, chain({ name: "ChannelSelection", channels: "2" }), "red").localPosition).toBeUndefined();
  });
  it("puts the user's shader and controls back instead of computing one", () => {
    const keep = { shader: "void main(){emitRGB(vec3(1.0));}", shaderControls: { normalized: { range: [0.2, 0.8] } } };
    expect(predictionLayer(served("mito"), chain(sigmoid), "green", keep)).toMatchObject(keep);
    expect(predictionLayer(served("mito"), chain(sigmoid), "green", { shader: "None" }).shader).toBe(predictionShader("green", [0, 1], "float32"));
  });
  it("colours models by their place in the submit order", () => {
    const layers = ["a", "b", "c"].map((name, i) => predictionLayer(served(name), chain(sigmoid), PREDICTION_COLORS[i % PREDICTION_COLORS.length]));
    expect(layers.map((l) => /color\(default="([^"]+)"\)/.exec(l.shader as string)?.[1])).toEqual(["red", "green", "blue"]);
  });
});

describe("dimensionsOf and centrePosition (bootstrap.raw_dimensions)", () => {
  it("gives the finest level's voxel size in metres per axis, as CoordinateSpace(units='nm') emits it", () => {
    // neuroglancer.CoordinateSpace(names=["z","y","x"], units="nm", scales=[8,8,8]).to_json()
    expect(dimensionsOf(info)).toEqual({ z: [8e-9, "m"], y: [8e-9, "m"], x: [8e-9, "m"] });
    // scales=[4, 4.5, 8]
    const mixed = { ...info, levels: [{ shape: [10, 10, 10], voxel: [4, 4.5, 8], origin: [0, 0, 0] }] };
    expect(dimensionsOf(mixed)).toEqual({ z: [4e-9, "m"], y: [4.5e-9, "m"], x: [8e-9, "m"] });
  });
  it("puts the position at the volume's centre, in voxels, the origin included", () => {
    expect(centrePosition(info)).toEqual([250, 500, 1000]);
    const offset = { ...info, levels: [{ shape: [500, 1000, 2000], voxel: [8, 8, 8], origin: [-4, -4, 800] }] };
    expect(centrePosition(offset)).toEqual([249.5, 499.5, 1100]);
  });
  it("skips a channel axis, whether or not the levels' arrays cover it", () => {
    const apart = { ...info, channels: 2, axes: [{ name: "c", unit: "" }, ...info.axes] };
    expect(dimensionsOf(apart)).toEqual(dimensionsOf(info));
    expect(centrePosition(apart)).toEqual([250, 500, 1000]);
    const together = {
      ...apart,
      levels: [{ shape: [2, 500, 1000, 2000], voxel: [1, 8, 8, 8], origin: [0, 0, 0, 0] }],
    };
    expect(dimensionsOf(together)).toEqual(dimensionsOf(info));
    expect(centrePosition(together)).toEqual([250, 500, 1000]);
  });
  it("reads axes named with their full unit", () => {
    const named = { ...info, axes: [{ name: "z", unit: "nanometer" }, { name: "y", unit: "nanometer" }, { name: "x", unit: "nanometer" }] };
    expect(dimensionsOf(named)).toEqual(dimensionsOf(info));
  });
});

describe("initialState (bootstrap.new_viewer)", () => {
  it("is the raw's dimensions, the centre, one voxel per pixel, the layers in order, one panel", () => {
    const raw = rawLayer(DATASET, info);
    const model = predictionLayer(served("mito"), chain(sigmoid), "red");
    const state = initialState(info, { data: raw, mito: model });
    expect(state).toEqual({
      gpuMemoryLimit: GPU_MEMORY_LIMIT,
      dimensions: { z: [8e-9, "m"], y: [8e-9, "m"], x: [8e-9, "m"] },
      position: [250, 500, 1000],
      crossSectionScale: 1,
      layers: [{ ...raw, name: "data" }, { ...model, name: "mito" }],
      layout: "xy",
    });
    expect(GPU_MEMORY_LIMIT).toBe(2_000_000_000); // bootstrap.GPU_MEMORY_LIMIT
    expect(DEFAULT_LAYOUT).toBe("xy");
  });
  it("names the layers by their keys, as Neuroglancer's array of layers wants", () => {
    expect(layerList({ mine: { type: "image", name: "other", source: "zarr3://a" } })).toEqual([{ type: "image", name: "mine", source: "zarr3://a" }]);
  });
  it("takes another layout when asked", () => {
    expect(initialState(info, {}, { layout: "4panel" }).layout).toBe("4panel");
  });
});

describe("stateWithoutInfo", () => {
  it("is the raw layer alone, the geometry left to Neuroglancer", () => {
    expect(stateWithoutInfo(DATASET)).toEqual({
      gpuMemoryLimit: GPU_MEMORY_LIMIT,
      layers: [{ type: "image", name: "data", source: `zarr://${DATASET}` }],
      layout: "xy",
    });
    const raw = rawLayer(DATASET, info);
    expect(stateWithoutInfo(DATASET, raw).layers).toEqual([{ ...raw, name: "data" }]);
  });
});
