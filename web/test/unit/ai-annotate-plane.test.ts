// From a planned plane to the image a model is sent (src/ai-annotate/plane.ts
// readPlane) against cellmap_flow/ai_annotate/pipeline.py: the raw read
// through a fake Inference whose readRaw records the box asked and answers a
// ramp, so the checks are where the read lands (the same physical area as the
// write box, with the level's origin taken as voxel 0's centre), how the
// plane is laid out (as the panel draws it) and the zeros past the level. The
// parity fixture checks the resampling, the mask put on the write box and the
// ids against the Python's own numbers, and the route tests
// (backend-ai-annotate.test.ts) run a plane all the way to labels.
import { describe, expect, it } from "vitest";

import type { DatasetInfo } from "../../src/contracts";
import { planPlane } from "../../src/ai-annotate/geometry";
import { readPlane } from "../../src/ai-annotate/plane";
import type { DepthAxis } from "../../src/ai-annotate/types";

/** A dataset of one level of `shape` voxels at 4 nm whose voxel 0's centre is at `origin`. */
function dataset(shape: number[], origin = [2, 2, 2], dtype = "uint8"): DatasetInfo {
  return {
    url: "https://example.org/raw.zarr", dtype, channels: 1,
    axes: [{ name: "z", unit: "nanometer" }, { name: "y", unit: "nanometer" }, { name: "x", unit: "nanometer" }],
    levels: [{ shape, voxel: [4, 4, 4], origin }],
  };
}

/** An 8 nm annotation volume over a 4 nm raw, voxel 0's centre at `offset`. */
function volume(offset = [4, 4, 4], output = [8, 8, 8], input = [4, 4, 4]) {
  return { output_voxel_size: output, input_voxel_size: input, dataset_offset_nm: offset };
}

/** The raw voxel (z, y, x) holds this, never 0, exact in float32. */
const rawValue = (z: number, y: number, x: number) => ((z * 7 + y) * 300 + x) % 60000 + 1;

/** An Inference's readRaw over a level of `shape` whose voxel (z, y, x) is
 * rawValue(z, y, x), recording every box asked and refusing one outside the
 * level, as the engine's reader would. */
function fakeInfer(shape: number[]) {
  const reads: { level: number; lo: number[]; hi: number[] }[] = [];
  return {
    reads,
    async readRaw(level: number, lo: number[], hi: number[]): Promise<ArrayBuffer> {
      reads.push({ level, lo: [...lo], hi: [...hi] });
      for (let a = 0; a < 3; a++) {
        if (lo[a] < 0 || hi[a] > shape[a] || hi[a] <= lo[a]) throw new Error(`a box ${lo.join(",")}..${hi.join(",")} is not within the level ${shape.join("×")}`);
      }
      const out = new Uint16Array((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]));
      let k = 0;
      for (let z = lo[0]; z < hi[0]; z++) for (let y = lo[1]; y < hi[1]; y++) for (let x = lo[2]; x < hi[2]; x++) out[k++] = rawValue(z, y, x);
      return out.buffer;
    },
  };
}

const LEVEL = [100, 200, 300];  // 4 nm voxels: 400 × 800 × 1200 nm
const VOLUME_SHAPE = [50, 100, 150];  // 8 nm voxels over the same extent

describe("readPlane", () => {
  for (const [origin, offset] of [[[2, 2, 2], [4, 4, 4]], [[102, 102, 102], [104, 104, 104]]]) {
    it(`reads the raw over the write box's physical area (level origin ${origin.join(",")}, volume offset ${offset.join(",")})`, async () => {
      // The level's corner and the volume's corner coincide (both 0, or both
      // 100): a plan's write box in 8 nm voxels and its raw read in 4 nm
      // voxels must cover the same nm.
      const levelCorner = origin[0] - 2, volumeCorner = offset[0] - 4;
      const data = dataset(LEVEL, origin);
      for (const depthAxis of [0, 1, 2] as DepthAxis[]) {
        const point = [volumeCorner + 201, volumeCorner + 403, volumeCorner + 606];  // in volume voxel (25, 50, 75)
        const plan = planPlane(volume(offset), VOLUME_SHAPE, point, depthAxis, 16);
        const infer = fakeInfer(LEVEL);
        const plane = await readPlane(infer, data, volume(offset), plan);
        expect(infer.reads.length).toBe(1);
        const { level, lo, hi } = infer.reads[0];
        expect(level).toBe(0);
        for (let a = 0; a < 3; a++) {
          if (a === depthAxis) {
            // The one raw voxel that holds the point.
            expect(hi[a] - lo[a]).toBe(1);
            expect(levelCorner + lo[a] * 4).toBeLessThanOrEqual(point[a]);
            expect(levelCorner + hi[a] * 4).toBeGreaterThan(point[a]);
          } else {
            expect(levelCorner + lo[a] * 4, `axis ${a} start`).toBe(volumeCorner + plan.write_lo[a] * 8);
            expect(levelCorner + hi[a] * 4, `axis ${a} end`).toBe(volumeCorner + plan.write_hi[a] * 8);
          }
        }
        expect([plane.rows, plane.cols]).toEqual(plan.image_shape);
        expect(plane.data).toBeInstanceOf(Float32Array);
      }
    });
  }

  it("lays the plane out as the panel draws it: XY with x leftwards, XZ and YZ transposed", async () => {
    const data = dataset(LEVEL, [2, 2, 2], "uint16");  // the fake answers uint16 values
    const point = [201, 403, 606];
    // XY (depth z): rows y, cols x running leftwards.
    let infer = fakeInfer(LEVEL);
    let plane = await readPlane(infer, data, volume(), planPlane(volume(), VOLUME_SHAPE, point, 0, 16));
    let { lo, hi } = infer.reads[0];
    expect([plane.rows, plane.cols]).toEqual([32, 32]);
    for (const [r, c] of [[0, 0], [5, 31], [31, 7]]) expect(plane.data[r * 32 + c]).toBe(rawValue(lo[0], lo[1] + r, hi[2] - 1 - c));
    // XZ (depth y): rows x, cols z.
    infer = fakeInfer(LEVEL);
    plane = await readPlane(infer, data, volume(), planPlane(volume(), VOLUME_SHAPE, point, 1, 16));
    ({ lo } = infer.reads[0]);
    for (const [r, c] of [[0, 0], [5, 31], [31, 7]]) expect(plane.data[r * 32 + c]).toBe(rawValue(lo[0] + c, lo[1], lo[2] + r));
    // YZ (depth x): rows y, cols z.
    infer = fakeInfer(LEVEL);
    plane = await readPlane(infer, data, volume(), planPlane(volume(), VOLUME_SHAPE, point, 2, 16));
    ({ lo } = infer.reads[0]);
    for (const [r, c] of [[0, 0], [5, 31], [31, 7]]) expect(plane.data[r * 32 + c]).toBe(rawValue(lo[0] + c, lo[1] + r, lo[2]));
  });

  it("clips the read to the level and leaves zeros past it", async () => {
    // The volume is padded past the level (to whole chunks): a plane near
    // the far x edge reads to the level's end and is zero beyond, on the
    // left of the XY image where x runs leftwards.
    const padded = [50, 100, 160];  // 1280 nm in x over a level of 1200
    const point = [201, 403, 1196];  // x voxel 149
    const plan = planPlane(volume(), padded, point, 0, 16);
    expect(plan.write_hi[2]).toBe(157);
    const infer = fakeInfer(LEVEL);
    const plane = await readPlane(infer, dataset(LEVEL, [2, 2, 2], "uint16"), volume(), plan);
    expect(infer.reads[0].hi[2]).toBe(300);
    expect(infer.reads[0].lo[2]).toBe(141 * 2);
    expect([plane.rows, plane.cols]).toEqual([32, 32]);
    // 314 − 300 = 14 raw columns past the level, mirrored to the left.
    for (let c = 0; c < 14; c++) expect(plane.data[5 * 32 + c], `col ${c}`).toBe(0);
    expect(plane.data[5 * 32 + 14]).toBe(rawValue(infer.reads[0].lo[0], infer.reads[0].lo[1] + 5, 299));
  });
});
