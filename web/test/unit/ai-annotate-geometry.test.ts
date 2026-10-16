// Planning the plane an AI annotation reads and the voxels it writes
// (src/ai-annotate/geometry.ts) against cellmap_flow/ai_annotate/geometry.py,
// with the arithmetic done by hand as tests/ai_annotate/test_geometry.py does
// it. The volumes here have their voxel 0's corner at the world origin: an
// 8 nm volume's dataset_offset_nm is 4 (voxel 0's centre), so annotation
// voxel i covers [8i, 8i + 8) nm and the expectations are plain. The parity
// fixture (test/parity/ai-annotate.test.ts) checks the same functions
// against the Python's own numbers.
import { describe, expect, it } from "vitest";

import {
  clickPx, formatG, fromScreen, planeAxes, planeName, planPlane, rawBox, roundHalfEven, screenShape, snapIntegral, toScreen, writeShape,
} from "../../src/ai-annotate/geometry";
import { AIAnnotateError, type DepthAxis, MAX_IMAGE_SIDE } from "../../src/ai-annotate/types";

function volume(output = [8, 8, 8], input = [4, 4, 4], offset?: number[]) {
  return { output_voxel_size: output, input_voxel_size: input, dataset_offset_nm: offset ?? output.map((v) => v / 2) };
}

const SHAPE = [100, 200, 300];
const AXES: DepthAxis[] = [0, 1, 2];

describe("numpy's and Python's arithmetic", () => {
  it("snapIntegral makes near-whole numbers whole and leaves the rest", () => {
    expect(snapIntegral(511.9999999)).toBe(512);
    expect(snapIntegral(512.0000001)).toBe(512);
    expect(snapIntegral(511.5)).toBe(511.5);
    expect(snapIntegral(1e9 + 100)).toBe(1e9 + 100);  // relative tolerance: 100 away from a billion is not near
    expect(snapIntegral(0)).toBe(0);
    expect(snapIntegral(-3.0000000001)).toBe(-3);
  });

  it("roundHalfEven rounds halves to the even neighbour as np.round does", () => {
    expect(roundHalfEven(500.5)).toBe(500);
    expect(roundHalfEven(501.5)).toBe(502);
    expect(roundHalfEven(0.5)).toBe(0);
    expect(roundHalfEven(1.5)).toBe(2);
    expect(roundHalfEven(2.4)).toBe(2);
    expect(roundHalfEven(2.6)).toBe(3);
    expect(roundHalfEven(-0.5) === 0).toBe(true);  // np.round gives -0.0; the sign of zero matters nowhere here
    expect(roundHalfEven(-1.5)).toBe(-2);
    expect(roundHalfEven(7)).toBe(7);
  });

  it("formatG spells numbers as Python's :g does", () => {
    const cases: [number, string][] = [
      [400, "400"], [4.5, "4.5"], [0.5, "0.5"], [-0.5, "-0.5"], [0, "0"], [100000, "100000"], [1000000, "1e+06"], [12345678, "1.23457e+07"],
      [0.0001, "0.0001"], [0.00001, "1e-05"], [803, "803"], [1206.25, "1206.25"], [2400, "2400"], [0.1 + 0.2, "0.3"], [1e-7, "1e-07"],
    ];
    for (const [value, text] of cases) expect(formatG(value), String(value)).toBe(text);
    // .3g, the prompt's resolution sentence: 8 says "8", 3.5 "3.5", not "4".
    expect(formatG(8, 3)).toBe("8");
    expect(formatG(3.5, 3)).toBe("3.5");
    expect(formatG(12.3456, 3)).toBe("12.3");
    expect(formatG(1024, 3)).toBe("1.02e+03");
    expect(formatG(0.5, 3)).toBe("0.5");
    expect(formatG(Number.NaN)).toBe("nan");
    expect(formatG(Number.POSITIVE_INFINITY)).toBe("inf");
  });
});

describe("screen layout", () => {
  // A 2 × 3 plane in z, y, x order: rows the earlier axis, cols the later.
  const plane = Int32Array.from([0, 1, 2, 3, 4, 5]);

  it("the XY plane (depth axis z) keeps y down and runs x leftwards", () => {
    const screen = toScreen(plane, 2, 3, 0);
    expect([screen.rows, screen.cols]).toEqual([2, 3]);
    expect(Array.from(screen.data)).toEqual([2, 1, 0, 5, 4, 3]);
    expect(screenShape([2, 3], 0)).toEqual([2, 3]);
  });

  it("the XZ and YZ planes are drawn transposed: the later axis down", () => {
    for (const axis of [1, 2] as DepthAxis[]) {
      const screen = toScreen(plane, 2, 3, axis);
      expect([screen.rows, screen.cols]).toEqual([3, 2]);
      expect(Array.from(screen.data)).toEqual([0, 3, 1, 4, 2, 5]);
      expect(screenShape([2, 3], axis)).toEqual([3, 2]);
    }
  });

  it("fromScreen undoes toScreen on every axis and keeps the array's kind", () => {
    for (const axis of AXES) {
      const screen = toScreen(plane, 2, 3, axis);
      const back = fromScreen(screen.data, screen.rows, screen.cols, axis);
      expect([back.rows, back.cols]).toEqual([2, 3]);
      expect(back.data).toBeInstanceOf(Int32Array);
      expect(Array.from(back.data)).toEqual(Array.from(plane));
    }
    const bytes = toScreen(Uint8Array.from([1, 2, 3, 4]), 2, 2, 0);
    expect(bytes.data).toBeInstanceOf(Uint8Array);
  });

  it("refuses a plane whose length is not rows × cols, and an axis that is not 0, 1 or 2", () => {
    expect(() => toScreen(plane, 2, 2, 0)).toThrow(/2×2/);
    expect(() => fromScreen(plane, 3, 3, 1)).toThrow(/3×3/);
    expect(() => screenShape([2, 3], 3 as DepthAxis)).toThrow(/depth_axis/);
  });
});

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

  it("a point on a voxel boundary takes the voxel above", () => {
    const plan = planPlane(volume(), SHAPE, [80, 800, 1200], 0, 8);
    expect([plan.write_lo[0], plan.write_hi[0]]).toEqual([10, 11]);
  });

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

  it("input and output voxel sizes may differ", () => {
    // 512 annotation voxels of 8 nm, read at 4 nm: a 1024 px image, as is.
    let plan = planPlane(volume([8, 8, 8], [4, 4, 4]), [100, 2000, 2000], [400, 8000, 8000], 0, 512);
    expect(writeShape(plan)).toEqual([512, 512]);
    expect(plan.image_shape).toEqual([1024, 1024]);
    expect(plan.resolution_nm).toBe(4);
    // Read at 8 nm, an 8 nm volume: one pixel per annotation voxel.
    plan = planPlane(volume([8, 8, 8], [8, 8, 8]), [100, 2000, 2000], [400, 8000, 8000], 0, 512);
    expect(plan.image_shape).toEqual([512, 512]);
    expect(plan.resolution_nm).toBe(8);
  });

  it("a plane longer than 1024 px is downsampled", () => {
    // 1024 voxels of 8 nm at 4 nm would be 2048 px: sent at 8 nm instead.
    let plan = planPlane(volume(), [100, 2000, 2000], [400, 8000, 8000], 0, 1024);
    expect(writeShape(plan)).toEqual([1024, 1024]);
    expect(plan.raw_shape_nm.slice(1)).toEqual([8192, 8192]);
    expect(plan.image_shape).toEqual([MAX_IMAGE_SIDE, MAX_IMAGE_SIDE]);
    expect(plan.resolution_nm).toBe(8);
    // The longer side decides; the other keeps the same square pixels.
    plan = planPlane(volume(), [100, 2000, 2000], [400, 8000, 40], 0, 1024);
    expect(writeShape(plan)).toEqual([1024, 517]);
    expect(plan.image_shape).toEqual([1024, 517]);
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

  it("the field is clipped to the data's box too, and a point outside the data is refused", () => {
    // The data ends at x = 1210 nm; the volume runs on to 2400.
    const plan = planPlane(volume(), SHAPE, point, 0, 64, [[0, 0, 0], [800, 1600, 1210]]);
    expect(plan.write_lo[2]).toBe(Math.ceil((1206 - 256) / 8 - 0.5));
    expect(plan.write_hi[2]).toBe(Math.ceil(1210 / 8 - 0.5));
    expect(plan.raw_offset_nm[2] + plan.raw_shape_nm[2]).toBe(plan.write_hi[2] * 8);
    let caught: unknown;
    try {
      planPlane(volume(), SHAPE, point, 0, 64, [[0, 0, 0], [800, 1600, 1200]]);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(AIAnnotateError);
    expect((caught as AIAnnotateError).category).toBe("refused");
    expect((caught as AIAnnotateError).httpStatus).toBe(400);
    expect((caught as AIAnnotateError).userMessage).toBe(
      "The point (401, 803, 1206) nm is outside the data (z, y, x 0..800, 0..1600, 0..1200 nm): move the view onto the data and try again.",
    );
  });

  it("a point outside the volume is refused with the volume's extent", () => {
    for (const p of [[-1, 800, 1200], [400, 800, 300 * 8], [400, -0.5, 1200], [100 * 8 + 4, 800, 1200]]) {
      let caught: unknown;
      try {
        planPlane(volume(), SHAPE, p, 0, 64);
      } catch (e) {
        caught = e;
      }
      expect(caught, p.join(",")).toBeInstanceOf(AIAnnotateError);
      const e = caught as AIAnnotateError;
      expect(e.category).toBe("refused");
      expect(e.httpStatus).toBe(400);
      expect(e.userMessage).toContain("outside the annotation volume (z, y, x 0..800, 0..1600, 0..2400 nm)");
    }
    expect(() => planPlane(volume(), SHAPE, [400, -0.5, 1200], 0, 64)).toThrow(/\(400, -0\.5, 1200\)/);
  });

  it("bad arguments are plain errors, not refusals", () => {
    expect(() => planPlane(volume(), SHAPE, [400, 800, 1200], 3 as DepthAxis, 64)).toThrow(/depth_axis/);
    expect(() => planPlane(volume(), SHAPE, [400, 800, 1200], 0, 0)).toThrow(/crop_size_px/);
    expect(() => planPlane(volume(), SHAPE, [400, Number.NaN, 1200], 0, 64)).toThrow(/point_nm/);
    expect(() => planPlane(volume(), SHAPE, [400, 800], 0, 64)).toThrow(/point_nm/);
    for (const bad of [() => planPlane(volume(), SHAPE, [400, 800, 1200], 0, 0)]) {
      let caught: unknown;
      try {
        bad();
      } catch (e) {
        caught = e;
      }
      expect(caught).not.toBeInstanceOf(AIAnnotateError);
    }
  });
});

describe("rawBox", () => {
  const plan = planPlane(volume(), SHAPE, [50 * 8 + 1, 800, 1200], 0, 64);

  it("reads exactly the box's voxels on a grid the box lies on, and the voxel holding the point", () => {
    // A raw at 4 nm from the origin: the one input voxel the point is in (401 nm is in [400, 404)).
    const box = rawBox(plan, [4, 4, 4], [0, 0, 0]);
    expect(box.begin).toEqual([100, (800 - 256) / 4, (1200 - 256) / 4]);
    expect(box.count).toEqual([1, 128, 128]);
  });

  it("reads the raw voxel each pixel's centre is in when the grid is off the box's", () => {
    // The first pixel's centre is 546 nm in y.
    for (const [corner, begin] of [[[-1, -1, -1], [100, 136, 236]], [[1, 1, 1], [100, 136, 236]], [[3, 3, 3], [99, 135, 235]]]) {
      const box = rawBox(plan, [4, 4, 4], corner);
      expect(box.begin, corner.join(",")).toEqual(begin);
      expect(box.count).toEqual([1, 128, 128]);
    }
  });

  it("reads a coarser raw by its own voxels, at least one", () => {
    const box = rawBox(plan, [8, 8, 8], [0, 0, 0]);
    expect(box.begin).toEqual([50, 68, 118]);
    expect(box.count).toEqual([1, 64, 64]);
    const coarse = rawBox(plan, [16, 16, 16], [0, 0, 0]);
    expect(coarse.count[0]).toBe(1);
  });
});
