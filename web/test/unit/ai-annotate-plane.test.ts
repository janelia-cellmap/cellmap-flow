// From a planned plane to labels (src/ai-annotate/plane.ts) against
// cellmap_flow/ai_annotate/pipeline.py: the raw read through a fake
// Inference whose readRaw records the box asked and answers a ramp, so the
// checks are where the read lands (the same physical area as the write box,
// with the level's origin taken as voxel 0's centre), how the plane is laid
// out (as the panel draws it), the zeros past the level, the resize and its
// isUint8; then the request built from it, the mask put on the write box and
// the labels for the box. The parity fixture checks the resampling and the
// ids against the Python's own numbers.
import { describe, expect, it } from "vitest";

import type { DatasetInfo } from "../../src/contracts";
import { planPlane, toScreen, writeShape } from "../../src/ai-annotate/geometry";
import { ORGANELLES } from "../../src/ai-annotate/organelles";
import { buildRequest, labelsForBox, levelCornerNm, maskToWriteShape, rawExtentNm, readPlane, resizePlane } from "../../src/ai-annotate/plane";
import { AIAnnotateError, type DepthAxis, type PlanePlan } from "../../src/ai-annotate/types";

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
function fakeInfer(shape: number[], dtype: "uint8" | "uint16" = "uint16") {
  const reads: { level: number; lo: number[]; hi: number[] }[] = [];
  const Typed = dtype === "uint8" ? Uint8Array : Uint16Array;
  return {
    reads,
    async readRaw(level: number, lo: number[], hi: number[]): Promise<ArrayBuffer> {
      reads.push({ level, lo: [...lo], hi: [...hi] });
      for (let a = 0; a < 3; a++) {
        if (lo[a] < 0 || hi[a] > shape[a] || hi[a] <= lo[a]) throw new Error(`a box ${lo.join(",")}..${hi.join(",")} is not within the level ${shape.join("×")}`);
      }
      const out = new Typed((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]));
      let k = 0;
      for (let z = lo[0]; z < hi[0]; z++) for (let y = lo[1]; y < hi[1]; y++) for (let x = lo[2]; x < hi[2]; x++) out[k++] = value(z, y, x, dtype);
      return out.buffer;
    },
  };
}

const value = (z: number, y: number, x: number, dtype: string) => (dtype === "uint8" ? rawValue(z, y, x) % 256 : rawValue(z, y, x));

const LEVEL = [100, 200, 300];  // 4 nm voxels: 400 × 800 × 1200 nm
const VOLUME_SHAPE = [50, 100, 150];  // 8 nm voxels over the same extent

describe("levelCornerNm and rawExtentNm", () => {
  it("take a level's origin as voxel 0's centre", () => {
    expect(levelCornerNm({ origin: [2, 2, 2], voxel: [4, 4, 4] })).toEqual([0, 0, 0]);
    expect(levelCornerNm({ origin: [102, 6, -2], voxel: [4, 8, 4] })).toEqual([100, 2, -4]);
    expect(rawExtentNm(dataset(LEVEL), volume())).toEqual([[0, 0, 0], [400, 800, 1200]]);
    expect(rawExtentNm(dataset(LEVEL, [102, 102, 102]), volume())).toEqual([[100, 100, 100], [500, 900, 1300]]);
  });
});

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

  it("is uint8 as read when the dataset is uint8 and nothing was resized, else float32 values", async () => {
    const point = [201, 403, 606];
    let plane = await readPlane(fakeInfer(LEVEL, "uint8"), dataset(LEVEL, [2, 2, 2], "uint8"), volume(), planPlane(volume(), VOLUME_SHAPE, point, 0, 16));
    expect(plane.isUint8).toBe(true);
    plane = await readPlane(fakeInfer(LEVEL, "uint16"), dataset(LEVEL, [2, 2, 2], "uint16"), volume(), planPlane(volume(), VOLUME_SHAPE, point, 0, 16));
    expect(plane.isUint8).toBe(false);
  });

  it("resizes an anisotropic plane to square pixels, bilinear along the grown axis", async () => {
    // 16 nm z voxels over a raw of 8 nm z, 4 nm y and x: the XZ plane's z
    // axis is read at 8 nm and drawn at 4, so the image doubles it.
    const aniso = volume([8, 4, 4], [16, 8, 8], [8, 4, 4]);
    const data: DatasetInfo = { ...dataset([50, 200, 300]), levels: [{ shape: [50, 200, 300], voxel: [8, 4, 4], origin: [4, 2, 2] }] };
    const plan = planPlane(aniso, [25, 100, 150], [200, 403, 606], 1, 16);
    expect(writeShape(plan)).toEqual([8, 16]);
    expect(plan.image_shape).toEqual([32, 32]);
    const infer = fakeInfer([50, 200, 300], "uint8");
    const plane = await readPlane(infer, data, aniso, plan);
    const { lo, hi } = infer.reads[0];
    expect(hi[0] - lo[0]).toBe(16);  // 16 raw z voxels of 8 nm
    expect(hi[2] - lo[2]).toBe(32);
    expect([plane.rows, plane.cols]).toEqual([32, 32]);
    expect(plane.isUint8).toBe(false);  // resized, so float32 values now
    // Rows are x (unchanged), columns z grown from 16 to 32: the two pixels
    // over one raw voxel average towards their neighbours, so the middle of
    // a constant run is that value.
    const atRow = (r: number, c: number) => plane.data[r * 32 + c];
    expect(atRow(3, 0)).toBeCloseTo(value(lo[0], lo[1], lo[2] + 3, "uint8"), 4);  // the edge pixel: only its own voxel in reach
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

  it("refuses a dataset with channels", async () => {
    const data = { ...dataset(LEVEL), channels: 2 };
    await expect(readPlane(fakeInfer(LEVEL), data, volume(), planPlane(volume(), VOLUME_SHAPE, [201, 403, 606], 0, 16))).rejects.toThrow(/channels/);
  });
});

describe("resizePlane", () => {
  it("returns the plane itself at its own size", () => {
    const plane = { rows: 2, cols: 3, data: Float32Array.from([1, 2, 3, 4, 5, 6]) };
    expect(resizePlane(plane, 2, 3)).toBe(plane);
  });

  it("averages whole blocks when shrinking by a whole factor (the box filter)", () => {
    const plane = { rows: 2, cols: 4, data: Float32Array.from([1, 3, 10, 20, 5, 7, 30, 40]) };
    const out = resizePlane(plane, 1, 2);
    expect([out.rows, out.cols]).toEqual([1, 2]);
    expect(Array.from(out.data)).toEqual([4, 25]);
  });

  it("keeps a constant plane constant and interpolates linearly when growing", () => {
    const flat = resizePlane({ rows: 2, cols: 2, data: Float32Array.from([9, 9, 9, 9]) }, 5, 7);
    expect(Array.from(flat.data).every((v) => Math.abs(v - 9) < 1e-5)).toBe(true);
    // Two columns 0 and 100 grown to 4: PIL's bilinear puts 0, 25, 75, 100.
    const ramp = resizePlane({ rows: 1, cols: 2, data: Float32Array.from([0, 100]) }, 1, 4);
    expect(Array.from(ramp.data)).toEqual([0, 25, 75, 100]);
  });

  it("uses the triangle filter for a mixed resize (one axis shrinks, the other grows)", () => {
    // Shrinking 4 → 2 with BOX gives [2, 4] for [1, 3, 3, 5]; BILINEAR weighs
    // the neighbours too and gives something else, which marks the filter.
    const box = resizePlane({ rows: 1, cols: 4, data: Float32Array.from([1, 3, 3, 5]) }, 1, 2);
    expect(Array.from(box.data)).toEqual([2, 4]);
    const mixed = resizePlane({ rows: 1, cols: 4, data: Float32Array.from([1, 3, 3, 5]) }, 2, 2);
    expect(Array.from(mixed.data.slice(0, 2))).not.toEqual([2, 4]);
    expect(mixed.data[0]).toBeCloseTo(mixed.data[2], 6);  // the grown axis repeats the one row
  });
});

describe("buildRequest", () => {
  const plan = planPlane(volume(), VOLUME_SHAPE, [201, 403, 606], 0, 16);

  it("makes the grey RGBA image, the prompt with the resolution sent, the colour and the click", () => {
    const data = new Float32Array(32 * 32);
    for (let i = 0; i < data.length; i++) data[i] = i % 256;
    const request = buildRequest({ rows: 32, cols: 32, data, isUint8: true }, plan, ORGANELLES.mito);
    expect([request.image.width, request.image.height]).toEqual([32, 32]);
    expect(request.image.data.length).toBe(32 * 32 * 4);
    expect(request.image.data[4 * 100]).toBe(100);
    expect(request.image.data[4 * 100 + 1]).toBe(100);
    expect(request.image.data[4 * 100 + 3]).toBe(255);
    expect(request.prompt.startsWith("The image resolution is 4 nm/px. This is an EM image of cell(s). In EM, mitochondria appear as:")).toBe(true);
    expect(request.prompt.endsWith("never merge touching instances into one connected blob.")).toBe(true);
    expect(request.target_rgb).toEqual([255, 0, 0]);
    expect(request.click_px).toEqual([16, 16]);
  });

  it("puts the override in place of the catalog's text, and refuses a plane off the plan", () => {
    const data = new Float32Array(32 * 32);
    const request = buildRequest({ rows: 32, cols: 32, data, isUint8: true }, plan, ORGANELLES.mito, "  Paint them.  ");
    expect(request.prompt).toContain(" Paint them. Where two individual instances");
    expect(request.prompt).not.toContain("appear as");
    expect(() => buildRequest({ rows: 16, cols: 32, data: new Float32Array(16 * 32), isUint8: true }, plan, ORGANELLES.mito)).toThrow(/The plane is \(16, 32\), the plan's image \(32, 32\)/);
  });
});

describe("maskToWriteShape", () => {
  /** A plan whose write box is `write` voxels (z, y, x) across `depthAxis` and whose image is `image`. */
  function planOf(depthAxis: DepthAxis, write: number[], image: [number, number]): PlanePlan {
    return {
      depth_axis: depthAxis, point_nm: [0, 0, 0], raw_offset_nm: [0, 0, 0], raw_shape_nm: [1, 1, 1], image_shape: image,
      input_voxel_size: [1, 1, 1], write_lo: [0, 0, 0], write_hi: write, resolution_nm: 1,
    };
  }

  it("samples each voxel's centre (the Python's test): an XZ plane drawn with z across", () => {
    // A 4 px image over 2 voxels: voxel 0 takes pixel 1 (its centre is at
    // 1.0, the pixel [1, 2)), voxel 1 pixel 3. The image's row is the box's column.
    const plan = planOf(1, [2, 1, 1], [1, 4]);
    expect(Array.from(maskToWriteShape(Uint8Array.from([1, 0, 0, 1]), plan))).toEqual([0, 1]);
    expect(Array.from(maskToWriteShape(Uint8Array.from([0, 1, 0, 0]), plan))).toEqual([1, 0]);
  });

  it("comes back to z, y, x order from the screen layout at the same size", () => {
    // XY: 2 rows (y) × 3 cols (x leftwards). Image pixel (0, 0) is x = 2.
    const plan = planOf(0, [1, 2, 3], [2, 3]);
    expect(Array.from(maskToWriteShape(Uint8Array.from([1, 0, 0, 0, 0, 1]), plan))).toEqual([0, 0, 1, 2, 0, 0]);  // two objects, numbered in the box's raster order
    // Values other than 1 count as set.
    expect(Array.from(maskToWriteShape(Uint8Array.from([255, 0, 0, 0, 0, 0]), plan))).toEqual([0, 0, 1, 0, 0, 0]);
  });

  it("refuses a mask that is not the plan's image", () => {
    expect(() => maskToWriteShape(new Uint8Array(5), planOf(0, [1, 2, 3], [2, 3]))).toThrow(/the plan's image 2×3/);
  });
});

describe("labelsForBox", () => {
  /** A one-voxel-thick box of `rows × cols` across `depthAxis`: its lo and hi. */
  const box = (depthAxis: DepthAxis, rows: number, cols: number): [number[], number[]] => {
    const hi = [rows, cols];
    hi.splice(depthAxis, 0, 1);
    return [[0, 0, 0], hi];
  };

  /** `mask` (the box's plane, z, y, x order) cut into objects by
   * maskToWriteShape, from an image of the box's own size. */
  const objectsOf = (mask: Uint8Array, depthAxis: DepthAxis, rows: number, cols: number): Int32Array => {
    const [lo, hi] = box(depthAxis, rows, cols);
    const image = toScreen(mask, rows, cols, depthAxis);
    return maskToWriteShape(image.data, {
      depth_axis: depthAxis, point_nm: [0, 0, 0], raw_offset_nm: [0, 0, 0], raw_shape_nm: [1, 1, 1], image_shape: [image.rows, image.cols],
      input_voxel_size: [1, 1, 1], write_lo: lo, write_hi: hi, resolution_nm: 1,
    });
  };

  for (const depthAxis of [0, 1, 2] as DepthAxis[]) {
    it(`gives each object an id and keeps corner-touching pieces apart (axis ${depthAxis})`, () => {
      const mask = new Uint8Array(6 * 8);
      for (const [r, c] of [[0, 0], [0, 1], [1, 0], [1, 1]]) mask[r * 8 + c] = 1;
      for (let r = 3; r < 5; r++) for (let c = 4; c < 7; c++) mask[r * 8 + c] = 1;
      mask[5 * 8 + 7] = 1;  // touches the second only at a corner: an object of its own
      const [lo, hi] = box(depthAxis, 6, 8);
      const labels = labelsForBox(objectsOf(mask, depthAxis, 6, 8), new Uint8Array(48), lo, hi, depthAxis, false);
      expect(labels).toBeInstanceOf(Uint8Array);
      for (let i = 0; i < 48; i++) if (!mask[i]) expect(labels[i]).toBe(1);
      expect(new Set([labels[0], labels[3 * 8 + 4], labels[5 * 8 + 7]])).toEqual(new Set([2, 3, 4]));
      expect(labels[0]).toBe(2);  // numbered in raster order of first pixel
      expect(labels[1 * 8 + 1]).toBe(2);
      expect(labels[4 * 8 + 6]).toBe(labels[3 * 8 + 4]);
    });
  }

  it("keeps a painted object's id", () => {
    const mask = new Uint8Array(36);
    for (let r = 1; r < 4; r++) for (let c = 1; c < 4; c++) mask[r * 6 + c] = 1;
    const existing = new Uint8Array(36);
    existing[1 * 6 + 1] = 7;  // the user painted part of the object as 7
    existing[5 * 6 + 5] = 9;  // and another object elsewhere
    const labels = labelsForBox(mask, existing, ...box(0, 6, 6), 0, false);
    for (let r = 1; r < 4; r++) for (let c = 1; c < 4; c++) expect(labels[r * 6 + c]).toBe(7);
  });

  it("counts up on an instance volume", () => {
    const mask = new Uint8Array(32);
    for (let r = 0; r < 4; r++) for (const c of [0, 1, 4, 5]) mask[r * 8 + c] = 1;
    const existing = new Uint16Array(32);
    existing[7] = 300;
    const labels = labelsForBox(objectsOf(mask, 1, 4, 8), existing, ...box(1, 4, 8), 1, true);
    expect(labels).toBeInstanceOf(Uint16Array);
    expect(new Set([labels[0], labels[4]])).toEqual(new Set([301, 302]));
  });

  it("an object continuing one in the plane beside keeps its id; a new one skips the ids beside", () => {
    // The plane before holds two objects, 2 on the left and 3 on the right.
    // This plane's mask continues only the right one, and adds a new one
    // where the plane before has nothing.
    const before = new Uint16Array(72).fill(1);
    const mask = new Uint8Array(72);
    for (let r = 1; r < 4; r++) {
      for (let c = 0; c < 3; c++) before[r * 12 + c] = 2;
      for (let c = 5; c < 8; c++) {
        before[r * 12 + c] = 3;
        mask[r * 12 + c] = 1;
      }
      for (let c = 9; c < 12; c++) mask[r * 12 + c] = 1;
    }
    for (const depthAxis of [0, 1, 2] as DepthAxis[]) {
      const labels = labelsForBox(objectsOf(mask, depthAxis, 6, 12), new Uint16Array(72), ...box(depthAxis, 6, 12), depthAxis, true, [before, null]);
      for (let r = 1; r < 4; r++) {
        for (let c = 5; c < 8; c++) expect(labels[r * 12 + c]).toBe(3);  // the same object, not a fresh id
        for (let c = 9; c < 12; c++) expect(labels[r * 12 + c]).toBe(4);  // a new object, not 2: that id is a different object right beside it
      }
      for (let i = 0; i < 72; i++) if (!mask[i]) expect(labels[i]).toBe(1);  // the ids beside are only consulted, never written
    }
  });

  it("a new object on a uint8 volume skips the ids beside", () => {
    const after = new Uint8Array(32);
    for (let r = 0; r < 4; r++) for (const c of [0, 1]) after[r * 8 + c] = 2;
    const mask = new Uint8Array(32);
    for (let r = 0; r < 4; r++) for (const c of [5, 6]) mask[r * 8 + c] = 1;
    const labels = labelsForBox(mask, new Uint8Array(32), ...box(0, 4, 8), 0, false, [null, after]);
    for (let r = 0; r < 4; r++) for (const c of [5, 6]) expect(labels[r * 8 + c]).toBe(3);
  });

  it("refuses planes beside, masks and boxes of another shape, and running out of ids", () => {
    const [lo, hi] = box(0, 4, 4);
    expect(() => labelsForBox(new Uint8Array(16).fill(1), new Uint8Array(16), lo, hi, 0, false, [new Uint8Array(20), null])).toThrow(/planes beside/);
    expect(() => labelsForBox(new Uint8Array(16).fill(1), new Uint8Array(20), lo, hi, 0, false)).toThrow(/box holds 20/);
    expect(() => labelsForBox(new Uint8Array(12).fill(1), new Uint8Array(16), lo, hi, 0, false)).toThrow(/box holds 12/);
    expect(() => labelsForBox(new Uint8Array(16).fill(1), new Uint8Array(16), [0, 0, 0], [2, 4, 4], 0, false)).toThrow(/one voxel thick/);
    // A uint8 box beside a plane that uses every id, over a voxel of it that
    // holds none (an object over an id beside would continue it): the new
    // object has no id to take.
    const full = new Uint8Array(255);
    for (let i = 1; i < 255; i++) full[i] = i + 1;
    const mask = new Uint8Array(255);
    mask[0] = 1;
    let caught: unknown;
    try {
      labelsForBox(mask, new Uint8Array(255), [0, 0, 0], [1, 1, 255], 0, false, [full, null]);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(AIAnnotateError);
    expect((caught as AIAnnotateError).httpStatus).toBe(409);
    expect((caught as AIAnnotateError).userMessage).toMatch(/^1 new objects do not fit in the volume's uint8 labels/);
  });
});
