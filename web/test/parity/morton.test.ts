// Parity of web/src/infer/morton.ts with cellmap_flow/post/postprocessors.py's
// MortonSegmentationRelabeling and with pymorton's interleave.
//
// The fixture (test/fixtures/morton.json) was written by the Python class
// itself (test/fixtures/make_fixtures.py): small label blocks [C, z, y, x]
// in uint8, uint32 and float32, and for each case the step as the dashboard
// sends it, the chunk's corner and voxel count, pymorton.interleave of the
// corner, and the uint64 block the Python returned (as decimal strings).
// `interleave` lists pymorton's value for more corners, including ones past
// the 10 bits per coordinate it keeps. Regenerate the fixture when the
// Python changes; tests/web/test_fixtures_fresh.py fails when it is stale.
import { describe, expect, it } from "vitest";

import type { Step } from "../../src/contracts";
import { interleave3, mortonIncrement } from "../../src/infer/morton";
import { type NdArray, applyPostprocessors, isSegmentation, outputChannels, outputDtype } from "../../src/infer/post";
import { type ArrayRecord, dtypeOf, loadFixture } from "./helpers";

interface Uint64Record {
  dtype: "uint64";
  shape: number[];
  data: string[];
}

interface MortonCase {
  id: string;
  steps: Step[];
  input: string;
  chunk_corner: number[];
  chunk_num_voxels: number;
  morton: number;
  output: Uint64Record;
  chain_output_dtype: string;
  chain_num_channels: number;
  chain_is_segmentation: boolean | null;
}

interface MortonFixture {
  seed: number;
  inputs: Record<string, ArrayRecord>;
  interleave: { corner: number[]; value: number }[];
  cases: MortonCase[];
}

const fixture = loadFixture<MortonFixture>("morton.json");

function block(name: string): NdArray {
  const record = fixture.inputs[name];
  if (!record) throw new Error(`no input ${name} in the fixture`);
  const shape = [...record.shape];
  switch (record.dtype) {
    case "uint8": return { data: Uint8Array.from(record.data), shape };
    case "uint32": return { data: Uint32Array.from(record.data), shape };
    case "float32": return { data: Float32Array.from(record.data), shape };
    default: throw new Error(`input dtype ${record.dtype} is not expected here`);
  }
}

describe("interleave3 matches pymorton.interleave", () => {
  for (const { corner, value } of fixture.interleave) {
    it(`(${corner.join(", ")}) -> ${value}`, () => {
      expect(interleave3(corner[0]!, corner[1]!, corner[2]!)).toBe(value);
    });
  }
});

describe("applyPostprocessors matches MortonSegmentationRelabeling", () => {
  for (const c of fixture.cases) {
    it(c.id, () => {
      expect(mortonIncrement(c.chunk_corner, c.chunk_num_voxels), "the increment").toBe(BigInt(c.chunk_num_voxels) * BigInt(c.morton));
      const nd = block(c.input);
      const out = applyPostprocessors(nd, c.steps, { chunkIndex: c.chunk_corner, chunkNumVoxels: c.chunk_num_voxels });
      expect(out.shape, `${c.id}: shape`).toEqual(c.output.shape);
      expect(dtypeOf(out.data), `${c.id}: dtype`).toBe("uint64");
      const data = out.data as BigUint64Array;
      expect(data.length, `${c.id}: length`).toBe(c.output.data.length);
      for (let i = 0; i < data.length; i++) {
        const e = BigInt(c.output.data[i]!);
        if (data[i] !== e) throw new Error(`${c.id}: element ${i} is ${data[i]}, the Python gave ${e}`);
      }
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
