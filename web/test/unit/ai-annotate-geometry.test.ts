// Planning the plane an AI annotation reads and the voxels it writes
// (src/ai-annotate/geometry.ts) against cellmap_flow/ai_annotate/geometry.py,
// with the arithmetic done by hand as tests/ai_annotate/test_geometry.py does
// it. The volumes here have their voxel 0's corner at the world origin: an
// 8 nm volume's dataset_offset_nm is 4 (voxel 0's centre), so annotation
// voxel i covers [8i, 8i + 8) nm and the expectations are plain. The parity
// fixture (test/parity/ai-annotate.test.ts) checks the same functions
// against the Python's own numbers.
import { describe, expect, it } from "vitest";

import { clickPx, planeAxes, planeName, planPlane, writeShape } from "../../src/ai-annotate/geometry";
import { AIAnnotateError, type DepthAxis } from "../../src/ai-annotate/types";

function volume(output = [8, 8, 8], input = [4, 4, 4], offset?: number[]) {
  return { output_voxel_size: output, input_voxel_size: input, dataset_offset_nm: offset ?? output.map((v) => v / 2) };
}

const SHAPE = [100, 200, 300];
const AXES: DepthAxis[] = [0, 1, 2];

describe("planPlane", () => {
  const point = [50 * 8 + 1, 100 * 8 + 3, 150 * 8 + 6];  // in voxel (50, 100, 150), off its centre

  for (const depthAxis of AXES) {
    it(`centres a plane on the point, one voxel thick along axis ${depthAxis}`, () => {
      const plan = planPlane(volume(), SHAPE, point, depthAxis, 64);
      const voxel = [50, 100, 150];
      for (let a = 0; a < 3; a++) {
        if (a === depthAxis) {
          expect([plan.write_lo[a], plan.write_hi[a]]).toEqual([voxel[a], voxel[a] + 1]);
        } else {
          // 64 voxels whose centres lie within 256 nm of the point.
          expect(plan.write_hi[a] - plan.write_lo[a]).toBe(64);
          for (let i = plan.write_lo[a]; i < plan.write_hi[a]; i++) expect(Math.abs((i + 0.5) * 8 - point[a])).toBeLessThanOrEqual(256);
        }
      }
      expect(writeShape(plan)).toEqual([64, 64]);
      // The raw is the same area at 4 nm, one input voxel thick at the point.
      for (const a of planeAxes(plan)) {
        expect(plan.raw_offset_nm[a]).toBe(plan.write_lo[a] * 8);
        expect(plan.raw_shape_nm[a]).toBe(64 * 8);
      }
      expect(plan.raw_shape_nm[depthAxis]).toBe(4);
      expect(plan.raw_offset_nm[depthAxis] + 2).toBe(point[depthAxis]);
      expect(plan.image_shape).toEqual([128, 128]);
      expect(plan.resolution_nm).toBe(4);
      expect(plan.input_voxel_size).toEqual([4, 4, 4]);
      expect(plan.point_nm).toEqual(point);
      expect(planeName(plan)).toBe(["XY", "XZ", "YZ"][depthAxis]);
      // The point's pixel: 4 nm pixels from the read's corner, laid out as the
      // panel draws the plane (rows, cols): y, x with x leftwards; x, z; y, z.
      const [down, across] = { 0: [1, 2], 1: [2, 0], 2: [1, 0] }[depthAxis];
      const row = Math.floor((point[down] - plan.raw_offset_nm[down]) / 4), col = Math.floor((point[across] - plan.raw_offset_nm[across]) / 4);
      expect(clickPx(plan)).toEqual([row, depthAxis === 0 ? 128 - 1 - col : col]);
    });
  }

  it("the volume's offset is voxel 0's centre", () => {
    // dataset_offset_nm 100: voxel 0 covers [96, 104), so 103 nm is in voxel 0 and 104 nm in voxel 1.
    const plan = planPlane(volume([8, 8, 8], [4, 4, 4], [100, 100, 100]), SHAPE, [103, 104 + 8 * 20, 104 + 8 * 30], 0, 4);
    expect(plan.write_lo).toEqual([0, 19, 29]);
    expect(plan.write_hi).toEqual([1, 23, 33]);
    expect(plan.raw_offset_nm.slice(1)).toEqual([96 + 19 * 8, 96 + 29 * 8]);
  });

  it("anisotropic voxels give a square field and square pixels", () => {
    // 16 nm z, 8 nm in y and x; the input at half each.
    const v = volume([16, 8, 8], [8, 4, 4]);
    const p = [50 * 16 + 8, 100 * 8 + 4, 150 * 8 + 4];
    const xz = planPlane(v, SHAPE, p, 1, 64);
    // 64 of the finer (x) voxels is 512 nm; that is 32 of the 16 nm z voxels.
    expect(writeShape(xz)).toEqual([32, 64]);
    expect(xz.raw_shape_nm).toEqual([512, 4, 512]);
    // Pixels are the finer input voxel size on both axes: the 8 nm z is drawn at 4.
    expect(xz.image_shape).toEqual([128, 128]);
    expect(xz.resolution_nm).toBe(4);
    const xy = planPlane(v, SHAPE, p, 0, 64);
    expect(writeShape(xy)).toEqual([64, 64]);
    expect(xy.raw_shape_nm).toEqual([8, 512, 512]);
    expect(xy.write_hi[0] - xy.write_lo[0]).toBe(1);
  });

  it("the write box is clipped at the volume edge, and the raw read with it", () => {
    // Voxel (50, 3, 297): 3 voxels from y's start, 2 from x's end.
    const plan = planPlane(volume(), SHAPE, [50 * 8 + 4, 3 * 8 + 4, 297 * 8 + 4], 0, 64);
    expect(plan.write_lo).toEqual([50, 0, 297 - 32]);
    expect(plan.write_hi).toEqual([51, 3 + 32, 300]);
    expect(writeShape(plan)).toEqual([35, 35]);
    expect(plan.raw_offset_nm.slice(1)).toEqual([0, (297 - 32) * 8]);
    expect(plan.raw_shape_nm.slice(1)).toEqual([35 * 8, 35 * 8]);
    expect(plan.image_shape).toEqual([70, 70]);
    // x runs leftwards in the XY plane, as Neuroglancer draws it.
    expect(clickPx(plan)).toEqual([Math.floor((3 * 8 + 4) / 4), 70 - 1 - Math.floor((32 * 8 + 4) / 4)]);
  });

  it("the field is clipped to the data's box too, and a point outside the data or the volume is refused", () => {
    // The data ends at x = 1210 nm; the volume runs on to 2400.
    const plan = planPlane(volume(), SHAPE, point, 0, 64, [[0, 0, 0], [800, 1600, 1210]]);
    expect(plan.write_lo[2]).toBe(Math.ceil((1206 - 256) / 8 - 0.5));
    expect(plan.write_hi[2]).toBe(Math.ceil(1210 / 8 - 0.5));
    expect(plan.raw_offset_nm[2] + plan.raw_shape_nm[2]).toBe(plan.write_hi[2] * 8);
    const refusals = [
      () => planPlane(volume(), SHAPE, point, 0, 64, [[0, 0, 0], [800, 1600, 1200]]),  // past the data's end in x
      () => planPlane(volume(), SHAPE, [-1, 800, 1200], 0, 64),  // before the volume in z
      () => planPlane(volume(), SHAPE, [400, 800, 300 * 8], 0, 64),  // at the volume's end in x
    ];
    for (const [i, refused] of refusals.entries()) {
      let caught: unknown;
      try {
        refused();
      } catch (e) {
        caught = e;
      }
      expect(caught, `case ${i}`).toBeInstanceOf(AIAnnotateError);
      expect((caught as AIAnnotateError).category).toBe("refused");
      expect((caught as AIAnnotateError).httpStatus).toBe(400);
    }
  });
});
