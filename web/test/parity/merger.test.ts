// Parity of web/src/infer/merger.ts with SimpleBlockwiseMerger in
// cellmap_flow/post/postprocessors.py, and of its equivalence map with
// neuroglancer.equivalence_map.EquivalenceMap.to_json().
//
// The fixture (test/fixtures/merger.json) was written by the Python class
// itself (test/fixtures/make_segment_fixtures.py): sequences of chunks of
// labels fed in order to one SimpleBlockwiseMerger with their corners, and
// after each chunk the uint64 block it returned and equivalences_json(),
// the sets of joined ids. Both are compared exactly after every chunk.
// tests/web/test_fixtures_fresh.py fails when the fixture is stale.
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Step } from "../../src/contracts";
import { makeMerger } from "../../src/infer/merger";
import { buildPostprocessors, isSegmentation, outputDtype, runPostprocessors } from "../../src/infer/post";
import { type ArrayRecord, loadFixture } from "./helpers";

interface MergerChunk {
  corner: number[];
  input: ArrayRecord;
  /** uint64: the values as decimal strings. */
  output: { dtype: string; shape: number[]; data: string[] };
  /** EquivalenceMap.to_json() after this chunk: sorted sets of decimal strings. */
  equivalences: string[][];
}

interface MergerCase {
  id: string;
  step: Step;
  chunks: MergerChunk[];
}

interface MergerFixture {
  seed: number;
  block: number[];
  cases: MergerCase[];
}

const fixture = loadFixture<MergerFixture>("merger.json");
const chunkNumVoxels = fixture.block.reduce((a, b) => a * b, 1);

function blockOf(chunk: MergerChunk) {
  expect(chunk.input.dtype).toBe("uint32");
  return { data: new Uint32Array(chunk.input.data), shape: [...chunk.input.shape] };
}

describe("SimpleBlockwiseMerger matches the Python merger chunk by chunk", () => {
  for (const c of fixture.cases) {
    it(c.id, () => {
      const { merger, process } = makeMerger(c.step, {});
      c.chunks.forEach((chunk, k) => {
        const label = `${c.id}: chunk ${k} at ${chunk.corner.join(",")}`;
        const out = process(blockOf(chunk), { chunkIndex: chunk.corner, chunkNumVoxels });
        expect(out.shape, `${label}: shape`).toEqual(chunk.output.shape);
        expect(out.data, `${label}: dtype`).toBeInstanceOf(BigUint64Array);
        expect(chunk.output.dtype).toBe("uint64");
        expect(out.data.length, `${label}: length`).toBe(chunk.output.data.length);
        for (let i = 0; i < out.data.length; i++) {
          if (out.data[i] !== BigInt(chunk.output.data[i])) {
            throw new Error(`${label}: element ${i} is ${out.data[i]}, the Python gave ${chunk.output.data[i]}`);
          }
        }
        expect(merger.equivalences.toJSON(), `${label}: equivalences`).toEqual(chunk.equivalences);
      });
    });
  }

  it("the fixture exercises what it says", () => {
    const final = Object.fromEntries(fixture.cases.map((c) => [c.id, c.chunks[c.chunks.length - 1].equivalences]));
    expect(final.grid.some((set) => set.length > 2), "an object through three chunks").toBe(true);
    expect(final.eroded_faces.length).toBeGreaterThan(0);
    expect(final.eroded_faces.length).toBeLessThan(final.grid.length);
    expect(final.eroded_twice_string).toEqual([]);
    expect(final.same_id_meets_itself.some((set) => set.length === 1), "an id that met only itself").toBe(true);
    expect(final.reverse_order).toEqual(final.grid);
    expect(final.chunk_seen_twice).toEqual(final.grid);
    expect(final.channel_1_string).toEqual(final.grid);
  });
});

describe("the built step reports the Python's sets to the layer's hook", () => {
  afterEach(() => vi.useRealTimers());

  for (const c of fixture.cases) {
    it(`${c.id}: the last report is the Python's final sets`, () => {
      vi.useFakeTimers();
      const reports: string[][][] = [];
      const processors = buildPostprocessors([c.step], { onEquivalences: (sets) => reports.push(sets) });
      for (const chunk of c.chunks) runPostprocessors(processors, blockOf(chunk), { chunkIndex: chunk.corner, chunkNumVoxels });
      vi.runAllTimers();  // the trailing report
      const final = c.chunks[c.chunks.length - 1].equivalences;
      if (final.length === 0) {
        expect(reports, "nothing was joined: nothing to report").toEqual([]);
      } else {
        expect(reports.length).toBeGreaterThan(0);
        expect(reports[reports.length - 1]).toEqual(final);
      }
    });
  }

  it("declares what the Python class declares", () => {
    const step: Step = { name: "SimpleBlockwiseMerger" };
    expect(outputDtype([step], "float32")).toBe("uint64");
    expect(isSegmentation([step])).toBe(true);
  });
});
