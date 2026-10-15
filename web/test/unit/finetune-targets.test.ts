// The target transforms (src/finetune/targets.ts) beyond what the parity
// suite checks against the Python classes (test/parity/targets.test.ts):
// how a model's manifest picks its transform, the offsets its channel names
// encode, the layout of a target on small hand-made chunks, negative
// offsets, uint16 ids, and the errors.
import { describe, expect, it } from "vitest";

import type { ModelEntry, ModelManifest } from "../../src/contracts";
import { affinityOffsets, affinityTargets, binaryTargets, targetTransformFor } from "../../src/finetune/targets";
import type { PaintedChunk } from "../../src/finetune/types";
import { entryFor } from "../../src/models/manifest";

/** A chunk of `shape` with `labels` (uint8 unless given as a Uint16Array). */
function chunk(shape: number[], labels: number[] | Uint16Array): PaintedChunk {
  return { index: [0, 0, 0], shape, labels: labels instanceof Uint16Array ? labels : Uint8Array.from(labels) };
}

/** A catalog entry whose manifest names `channels`. */
function entryWith(channels: string[]): ModelEntry {
  const manifest: ModelManifest = {
    model: "setup_t", task: "t", architecture: "", checkpoint: "", params: 0, channels,
    input_voxel_size_nm: [8, 8, 8], output_voxel_size_nm: [8, 8, 8], output: "logits", input_norm: [],
    io: { input: "input", output: "output" },
    variants: [{ variant: "fp32", input_size: [6, 6, 6], output_size: [2, 2, 2], file: "setup_t_fp32.onnx", file_mb: 1 }],
  };
  return entryFor("setup_t", manifest, manifest.variants[0]);
}

const UNIT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

describe("affinityOffsets", () => {
  it("reads the unit offset off each channel's _aff_z/y/x suffix, in the channels' order", () => {
    expect(affinityOffsets(["mito_aff_z", "mito_aff_y", "mito_aff_x"])).toEqual(UNIT);
    expect(affinityOffsets(["a_aff_x", "a_aff_z"])).toEqual([[0, 0, 1], [1, 0, 0]]);
    expect(affinityOffsets([])).toEqual([]);
  });

  it("gives a channel that names no offset null", () => {
    expect(affinityOffsets(["nuc"])).toEqual([null]);
    expect(affinityOffsets(["mito_aff_z", "mito_aff_y", "mito_aff_x", "lsd_0"])).toEqual([...UNIT, null]);
    expect(affinityOffsets(["mito_aff"])).toEqual([null]);  // no axis
    expect(affinityOffsets(["mito_aff_z_extra"])).toEqual([null]);  // the suffix must end the name
  });
});

describe("targetTransformFor", () => {
  it("gives an affinity model affinities along its channels' offsets", () => {
    const transform = targetTransformFor(entryWith(["mito_aff_z", "mito_aff_y", "mito_aff_x"]));
    // One slice, one row of three voxels: object 2, object 2, background.
    const out = transform(chunk([1, 1, 3], [2, 2, 1]));
    expect(out.channels).toBe(3);
    // z: a one-slice chunk pairs nothing along z. y: one row, nothing either.
    expect(Array.from(out.mask.subarray(0, 6))).toEqual([0, 0, 0, 0, 0, 0]);
    // x: voxel 0 with 1 (same object), voxel 1 with 2 (an object and background), voxel 2 has no neighbour.
    expect(Array.from(out.target.subarray(6, 9))).toEqual([1, 0, 0]);
    expect(Array.from(out.mask.subarray(6, 9))).toEqual([1, 1, 0]);
  });

  it("masks out the channels of an affinity model that name no offset, as the server masks its LSDs", () => {
    const transform = targetTransformFor(entryWith(["mito_aff_z", "mito_aff_y", "mito_aff_x", "lsd_0", "lsd_1"]));
    const out = transform(chunk([1, 1, 3], [2, 2, 1]));
    expect(out.channels).toBe(5);
    expect(Array.from(out.mask.subarray(6, 9))).toEqual([1, 1, 0]);  // x trains
    expect(Array.from(out.mask.subarray(9, 15))).toEqual([0, 0, 0, 0, 0, 0]);  // the LSDs do not
    expect(Array.from(out.target.subarray(9, 15))).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it("decides by the server's test, a channel containing _aff, and refuses one whose offsets no name says", () => {
    expect(() => targetTransformFor(entryWith(["mito_aff"]))).toThrow(/none is named _aff_z, _aff_y or _aff_x/);
    // "aff" without the underscore is not the server's test: a binary model.
    const out = targetTransformFor(entryWith(["affine"]))(chunk([1, 1, 2], [1, 2]));
    expect(Array.from(out.target)).toEqual([0, 1]);
  });

  it("gives every other model the binary target on each of its channels", () => {
    const out = targetTransformFor(entryWith(["nuc", "nuc_edge"]))(chunk([1, 1, 3], [0, 1, 2]));
    expect(out.channels).toBe(2);
    expect(Array.from(out.target)).toEqual([0, 0, 1, 0, 0, 1]);
    expect(Array.from(out.mask)).toEqual([0, 1, 1, 0, 1, 1]);
  });

  it("refuses a manifest without channels", () => {
    expect(() => targetTransformFor(entryWith([]))).toThrow(/names no output channels/);
  });
});

describe("binaryTargets", () => {
  it("is 1 for an object and masked where painted, the same on every channel", () => {
    const out = binaryTargets(3)(chunk([1, 2, 2], [0, 1, 2, 7]));
    expect(out.channels).toBe(3);
    expect(out.target).toBeInstanceOf(Float32Array);
    for (let c = 0; c < 3; c++) {
      expect(Array.from(out.target.subarray(c * 4, c * 4 + 4)), `channel ${c} target`).toEqual([0, 0, 1, 1]);
      expect(Array.from(out.mask.subarray(c * 4, c * 4 + 4)), `channel ${c} mask`).toEqual([0, 1, 1, 1]);
    }
  });

  it("takes uint16 ids", () => {
    const out = binaryTargets(1)(chunk([1, 1, 3], Uint16Array.from([0, 1, 40000])));
    expect(Array.from(out.target)).toEqual([0, 0, 1]);
    expect(Array.from(out.mask)).toEqual([0, 1, 1]);
  });

  it("refuses no channels and a chunk whose labels do not fill it", () => {
    expect(() => binaryTargets(0)).toThrow(/at least one channel/);
    expect(() => binaryTargets(1)(chunk([2, 2, 2], [1, 2, 3]))).toThrow(/8 voxels, but 3 labels/);
  });
});

describe("affinityTargets", () => {
  it("pairs each voxel with the one at the offset and writes the pair at the voxel", () => {
    // Two slices of one row: z pairs (0, y, x) with (1, y, x).
    const out = affinityTargets([[1, 0, 0]], 1)(chunk([2, 1, 3], [2, 2, 3, 2, 0, 3]));
    expect(Array.from(out.target)).toEqual([1, 0, 1, 0, 0, 0]);  // same object, unpainted partner, same object; slice 1 has no partner
    expect(Array.from(out.mask)).toEqual([1, 0, 1, 0, 0, 0]);
  });

  it("takes a negative offset: the first voxels along the axis have no partner", () => {
    const out = affinityTargets([[0, 0, -1]], 1)(chunk([1, 1, 3], [2, 2, 3]));
    // _offset_slices(-1): source x in [1, 3), partner one before.
    expect(Array.from(out.target)).toEqual([0, 1, 0]);
    expect(Array.from(out.mask)).toEqual([0, 1, 1]);
  });

  it("pairs nothing along an axis the offset does not fit in", () => {
    const out = affinityTargets(UNIT, 3)(chunk([1, 2, 2], [2, 2, 2, 2]));
    expect(Array.from(out.mask.subarray(0, 4))).toEqual([0, 0, 0, 0]);  // z on a one-slice chunk
    expect(Array.from(out.mask.subarray(4, 8))).toEqual([1, 1, 0, 0]);
    expect(Array.from(out.mask.subarray(8, 12))).toEqual([1, 0, 1, 0]);
    expect(Array.from(out.target.subarray(4, 12))).toEqual([1, 1, 0, 0, 1, 0, 1, 0]);
    expect(Array.from(affinityTargets([[0, 0, 5]], 1)(chunk([1, 1, 3], [2, 2, 2])).mask)).toEqual([0, 0, 0]);
  });

  it("leaves the channels past the offsets, and a channel with a null offset, masked out", () => {
    const out = affinityTargets([[0, 0, 1]], 3)(chunk([1, 1, 2], [2, 2]));
    expect(out.channels).toBe(3);
    expect(Array.from(out.target)).toEqual([1, 0, 0, 0, 0, 0]);
    expect(Array.from(out.mask)).toEqual([1, 0, 0, 0, 0, 0]);
    const withNull = affinityTargets([null, [0, 0, 1]], 2)(chunk([1, 1, 2], [2, 2]));
    expect(Array.from(withNull.target)).toEqual([0, 0, 1, 0]);
    expect(Array.from(withNull.mask)).toEqual([0, 0, 1, 0]);
  });

  it("tells background from objects and one object from another, with uint16 ids", () => {
    const out = affinityTargets([[0, 0, 1]], 1)(chunk([1, 1, 5], Uint16Array.from([1, 1, 300, 300, 301])));
    expect(Array.from(out.target)).toEqual([0, 0, 1, 0, 0]);  // background pairs are 0 even when equal
    expect(Array.from(out.mask)).toEqual([1, 1, 1, 1, 0]);
  });

  it("refuses more offsets than channels and an offset that is not three whole numbers", () => {
    expect(() => affinityTargets(UNIT, 2)).toThrow(/3 affinity offsets for 2 channel/);
    expect(() => affinityTargets([[1, 0]], 1)).toThrow(/three whole numbers/);
    expect(() => affinityTargets([[0.5, 0, 0]], 1)).toThrow(/three whole numbers/);
  });
});
