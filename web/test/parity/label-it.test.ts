// Parity of the browser's Label it tools with the server's, where the
// browser runs TypeScript of its own: a seed's mutex watershed
// (web/src/label-it/segment.ts mutexObjects) with
// cellmap_flow/dashboard/routes/finetune/view_labels.py's
// (segment_prediction, on mwatershed). The rest of a seed, and the ids,
// writes and undo of every Label it tool, are the Python both modes run
// (cellmap_flow/portable/label_it.py and fill.py), tested by pytest.
//
// The fixture (test/fixtures/label_it.json) was written by the Python
// functions themselves (test/fixtures/make_label_it_fixtures.py): affinities,
// as the browser's Python hands them to the watershed, segmented into
// objects. Objects match exactly. Regenerate the fixture when the Python
// changes; tests/web/test_fixtures_fresh.py fails when it is stale.
import { describe, expect, it } from "vitest";

import { mutexObjects } from "../../src/label-it/segment";
import { type ArrayRecord, expectArrayMatches, loadFixture } from "./helpers";

interface Fixture {
  seed: number;
  mutex_watershed: {
    id: string; output_class: string | null; offsets: number[][]; threshold: number; min_size: number;
    affinities: ArrayRecord; objects: ArrayRecord; count: number;
  }[];
}

const fixture = loadFixture<Fixture>("label_it.json");

describe("mutex_watershed", () => {
  for (const c of fixture.mutex_watershed) {
    it(c.id, () => {
      const affinities = { data: Float32Array.from(c.affinities.data), shape: c.affinities.shape };
      const objects = mutexObjects(affinities, c.offsets, c.threshold, c.min_size);
      expectArrayMatches(objects, c.objects, c.id);
      expect(Math.max(0, ...objects)).toBe(c.count);
    });
  }

  it("has objects to find, and cases that drop some by size", () => {
    expect(fixture.mutex_watershed.some((c) => c.count > 2)).toBe(true);
    expect(fixture.mutex_watershed.some((c) => c.min_size > 1)).toBe(true);
  });
});
