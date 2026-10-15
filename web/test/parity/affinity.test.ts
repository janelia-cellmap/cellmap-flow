// Parity of web/src/infer/affinity.ts with cellmap_flow/post/postprocessors.py's
// AffinityPostprocessor and cellmap_flow/post/segment.py's mutex_watershed
// (which runs mwatershed.agglom).
//
// The fixture (test/fixtures/affinity.json) was written by the Python classes
// themselves (test/fixtures/make_fixtures.py): blocks of affinities
// [C, z, y, x], and for each case the step as the dashboard sends it, what
// segment.mutex_watershed returned on its own (labels 1..n by first
// appearance, 0 background) and what the step returned for a chunk off the
// origin (the same labels plus chunk_num_voxels * pymorton.interleave of the
// corner), both uint64 as decimal strings. The seed keeps every edge weight
// distinct and nonzero and every fragment's mean affinity clear of the bias,
// so the labels must match exactly: the partition is mwatershed's, the
// numbering fastremap.renumber's. Regenerate the fixture when the Python
// changes; tests/web/test_fixtures_fresh.py fails when it is stale.
import { describe, expect, it } from "vitest";

import type { Step } from "../../src/contracts";
import { DEFAULT_BIAS, mutexWatershed, parseNeighborhood } from "../../src/infer/affinity";
import { mortonIncrement } from "../../src/infer/morton";
import { asNumber } from "../../src/infer/norms";
import { type NdArray, applyPostprocessors, isSegmentation, outputChannels, outputDtype } from "../../src/infer/post";
import { type ArrayRecord, dtypeOf, loadFixture } from "./helpers";

/** A uint64 array as the fixtures carry it: the values as decimal strings. */
interface Uint64Record {
  dtype: "uint64";
  shape: number[];
  data: string[];
}

/** A block too large to list: its little-endian C-order bytes, base64. */
interface PackedRecord {
  dtype: string;
  shape: number[];
  base64: string;
}

interface AffinityCase {
  id: string;
  steps: Step[];
  /** The name of the block in `inputs`. */
  input: string;
  bias: number;
  neighborhood: number[][];
  chunk_corner: number[];
  chunk_num_voxels: number;
  /** pymorton.interleave(*chunk_corner). */
  morton: number;
  segmentation: Uint64Record;
  output: Uint64Record;
  chain_output_dtype: string;
  chain_num_channels: number;
  chain_is_segmentation: boolean | null;
}

interface AffinityFixture {
  seed: number;
  inputs: Record<string, ArrayRecord | PackedRecord>;
  cases: AffinityCase[];
}

const fixture = loadFixture<AffinityFixture>("affinity.json");

/** The named input block as the typed array of its dtype. */
function block(name: string): NdArray {
  const record = fixture.inputs[name];
  if (!record) throw new Error(`no input ${name} in the fixture`);
  if ("base64" in record) {
    const bytes = Uint8Array.from(Buffer.from(record.base64, "base64"));  // a copy: aligned at 0
    if (record.dtype !== "float32") throw new Error(`packed ${record.dtype} is not expected here`);
    return { data: new Float32Array(bytes.buffer), shape: [...record.shape] };
  }
  if (record.dtype === "uint8") return { data: Uint8Array.from(record.data), shape: [...record.shape] };
  if (record.dtype === "float32") return { data: Float32Array.from(record.data), shape: [...record.shape] };
  throw new Error(`input dtype ${record.dtype} is not expected here`);
}

/** Every element of `actual` is the fixture's, read as a bigint. */
function expectUint64Matches(actual: BigUint64Array, expected: Uint64Record, label: string): void {
  expect(actual.length, `${label}: length`).toBe(expected.data.length);
  for (let i = 0; i < expected.data.length; i++) {
    const e = BigInt(expected.data[i]!);
    if (actual[i] !== e) throw new Error(`${label}: element ${i} is ${actual[i]}, the Python gave ${e}`);
  }
}

describe("the fixture", () => {
  it("has cases with objects and background in each", () => {
    expect(fixture.cases.length).toBeGreaterThan(0);
    for (const c of fixture.cases) {
      const ids = c.segmentation.data.map(BigInt);
      expect(ids.some((v) => v === 0n), `${c.id}: background`).toBe(true);
      expect(ids.some((v) => v !== 0n), `${c.id}: objects`).toBe(true);
    }
  });
});

describe("the step's parameters parse as the Python constructor's", () => {
  for (const c of fixture.cases) {
    it(c.id, () => {
      const step = c.steps[0]!;
      expect(parseNeighborhood(step.neighborhood), "neighborhood").toEqual(c.neighborhood);
      expect(asNumber(step.bias, DEFAULT_BIAS, "bias"), "bias").toBe(c.bias);
      expect(mortonIncrement(c.chunk_corner, c.chunk_num_voxels), "the Morton increment").toBe(BigInt(c.chunk_num_voxels) * BigInt(c.morton));
    });
  }
});

describe("mutexWatershed matches segment.mutex_watershed", () => {
  for (const c of fixture.cases) {
    it(c.id, () => {
      const nd = block(c.input);
      // Integer input is scaled by the step before the segmenter sees it; the
      // fixture's segmentation was computed on the scaled values.
      const integer = !(nd.data instanceof Float32Array);
      const data = integer ? Float64Array.from(nd.data as Uint8Array, (v) => v / 255) : nd.data;
      const labels = mutexWatershed({ data, shape: nd.shape }, c.neighborhood, c.bias);
      expect(c.segmentation.shape).toEqual(nd.shape.slice(1));
      expectUint64Matches(labels, c.segmentation, c.id);
    });
  }
});

describe("applyPostprocessors matches AffinityPostprocessor", () => {
  for (const c of fixture.cases) {
    it(c.id, () => {
      const nd = block(c.input);
      const out = applyPostprocessors(nd, c.steps, { chunkIndex: c.chunk_corner, chunkNumVoxels: c.chunk_num_voxels });
      expect(out.shape, `${c.id}: shape`).toEqual(c.output.shape);
      expect(dtypeOf(out.data), `${c.id}: dtype`).toBe("uint64");
      expectUint64Matches(out.data as BigUint64Array, c.output, c.id);
    });
  }
});

describe("the chain summaries match pipeline_spec.py", () => {
  for (const c of fixture.cases) {
    it(c.id, () => {
      const channels = block(c.input).shape[0]!;
      expect(outputDtype(c.steps, "float32"), "outputDtype").toBe(c.chain_output_dtype);
      expect(outputChannels(c.steps, channels), "outputChannels").toBe(c.chain_num_channels);
      expect(isSegmentation(c.steps) ?? null, "isSegmentation").toBe(c.chain_is_segmentation);
    });
  }
});
