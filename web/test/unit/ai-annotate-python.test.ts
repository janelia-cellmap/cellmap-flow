// The AI annotation's calls into cellmap_flow.portable (portable/
// ai_annotate_calls.py), here in node's Pyodide (test/unit/pyodide.ts): that
// each answers through the browser's wrappers (plane.ts, mask.ts, the job's
// accept), and what is the browser's own around them: where the raw read
// lands (a level's origin is voxel 0's centre, so the read covers the write
// box's physical area) and the zeros past the level. The maths is tested in
// pytest on the Python itself (tests/portable/test_ai_annotate.py); the route
// tests (backend-ai-annotate.test.ts) run a plane all the way to labels.
// Without Pyodide (no network for its wheels) these skip.
import { describe, expect, it, vi } from "vitest";

import { modelMask, stageMask } from "../../src/ai-annotate/mask";
import { AI_PYTHON, planPlane, readImage } from "../../src/ai-annotate/plane";
import type { DepthAxis, RgbaImage } from "../../src/ai-annotate/types";
import type { DatasetInfo } from "../../src/contracts";
import { portablePython } from "./pyodide";

// Pyodide, numpy, Pillow (for the resizes) and scipy (for the mask's objects)
// load once for the file.
vi.setConfig({ testTimeout: 120_000 });
const python = await portablePython();

/** A dataset of one uint8 level of `shape` voxels at 4 nm whose voxel 0's centre is at `origin`. */
function dataset(shape: number[], origin = [2, 2, 2]): DatasetInfo {
  return {
    url: "https://example.org/raw.zarr", dtype: "uint8", channels: 1,
    axes: [{ name: "z", unit: "nanometer" }, { name: "y", unit: "nanometer" }, { name: "x", unit: "nanometer" }],
    levels: [{ shape, voxel: [4, 4, 4], origin }],
  };
}

/** An 8 nm annotation volume of `shape` voxels over a 4 nm raw, voxel 0's centre at `offset`. */
function volume(shape: number[], offset = [4, 4, 4]) {
  return { output_voxel_size: [8, 8, 8], input_voxel_size: [4, 4, 4], dataset_offset_nm: offset, dataset_shape_voxels: shape };
}

/** The raw voxel (z, y, x) holds this, never 0. */
const rawValue = (z: number, y: number, x: number) => ((z * 7 + y) * 3 + x) % 250 + 1;

/** An Inference with the portable Python and a readRaw over a level of
 * `shape` whose voxel (z, y, x) is rawValue(z, y, x), recording every box
 * asked and refusing one outside the level, as the engine's reader would. */
function fakeInfer(shape: number[]) {
  const reads: { level: number; lo: number[]; hi: number[] }[] = [];
  return {
    reads,
    python: python!,
    async readRaw(level: number, lo: number[], hi: number[]): Promise<ArrayBuffer> {
      reads.push({ level, lo: [...lo], hi: [...hi] });
      for (let a = 0; a < 3; a++) {
        if (lo[a] < 0 || hi[a] > shape[a] || hi[a] <= lo[a]) throw new Error(`a box ${lo.join(",")}..${hi.join(",")} is not within the level ${shape.join("×")}`);
      }
      const out = new Uint8Array((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]));
      let k = 0;
      for (let z = lo[0]; z < hi[0]; z++) for (let y = lo[1]; y < hi[1]; y++) for (let x = lo[2]; x < hi[2]; x++) out[k++] = rawValue(z, y, x);
      return out.buffer;
    },
  };
}

const LEVEL = [100, 200, 300];  // 4 nm voxels: 400 × 800 × 1200 nm
const VOLUME_SHAPE = [50, 100, 150];  // 8 nm voxels over the same extent

/** A grey image of `rows` × `cols` with the pixels `painted` says painted `rgb`, as a model might answer. */
function reply(rows: number, cols: number, painted: (r: number, c: number) => boolean, rgb = [255, 0, 0]): RgbaImage {
  const data = new Uint8ClampedArray(rows * cols * 4);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const j = (r * cols + c) * 4;
    const [red, green, blue] = painted(r, c) ? rgb : [40, 40, 40];
    data.set([red, green, blue, 255], j);
  }
  return { width: cols, height: rows, data };
}

describe.skipIf(!python)("the AI annotation's portable Python", () => {
  for (const [origin, offset] of [[[2, 2, 2], [4, 4, 4]], [[102, 102, 102], [104, 104, 104]]]) {
    it(`plans a plane whose raw read covers its write box (level origin ${origin.join(",")}, volume offset ${offset.join(",")})`, async () => {
      // The level's corner and the volume's corner coincide (both 0, or both
      // 100): a write box in 8 nm voxels and its raw read in 4 nm voxels
      // must cover the same nm.
      const levelCorner = origin[0] - 2, volumeCorner = offset[0] - 4;
      for (const depthAxis of [0, 1, 2] as DepthAxis[]) {
        const point = [volumeCorner + 201, volumeCorner + 403, volumeCorner + 606];  // in volume voxel (25, 50, 75)
        const infer = fakeInfer(LEVEL);
        const planned = await planPlane(infer, dataset(LEVEL, origin), volume(VOLUME_SHAPE, offset), point, depthAxis, 16);
        const image = await readImage(infer, dataset(LEVEL, origin), planned);
        const { plan } = planned;
        const { lo, hi } = infer.reads[0];
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
        expect([image.height, image.width]).toEqual(plan.image_shape);
        expect(planned.write_shape).toEqual([16, 16]);
      }
    });
  }

  it("sends the plane as the panel draws it, stops it at the data's edge, and refuses a point outside the volume", async () => {
    // The volume is padded past the level (to whole chunks): a plane near the
    // far x edge stops where the data does (the level's extent, rawExtentNm),
    // 9 voxels from 141. In the XY image x runs leftwards; a uint8 plane on
    // the image's own grid is sent as it is, so the image is the raw's values.
    const infer = fakeInfer(LEVEL);
    const planned = await planPlane(infer, dataset(LEVEL), volume([50, 100, 160]), [201, 403, 1196], 0, 16);
    expect([planned.plan.write_lo[2], planned.plan.write_hi[2]]).toEqual([141, 150]);
    const image = await readImage(infer, dataset(LEVEL), planned);
    const { lo, hi } = infer.reads[0];
    expect([lo[2], hi[2]]).toEqual([282, 300]);
    expect([image.height, image.width]).toEqual([32, 18]);
    const grey = (r: number, c: number) => image.data[(r * image.width + c) * 4];
    for (const c of [0, 9, 17]) expect(grey(5, c), `col ${c}`).toBe(rawValue(lo[0], lo[1] + 5, 299 - c));
    expect(image.data[3]).toBe(255);
    await expect(planPlane(infer, dataset(LEVEL), volume(VOLUME_SHAPE), [-1, 403, 606], 0, 16)).rejects.toMatchObject({ category: "refused", httpStatus: 400 });
  });

  it("takes the mask out of a reply at twice the size sent, lined up with the plane", async () => {
    // The plane is 32 × 24, sent padded to 32 × 32; the model answered 64 ×
    // 64 with the plane's left half red (in its own pixels, the left 24).
    const answer = reply(64, 64, (_r, c) => c < 24);
    const { mask, model_image } = await modelMask(python!, answer, { width: 32, height: 32 }, { width: 24, height: 32 }, [255, 0, 0]);
    expect([model_image.width, model_image.height, mask.length]).toEqual([24, 32, 24 * 32]);
    expect(mask[10 * 24 + 5]).toBe(1);
    expect(mask[10 * 24 + 20]).toBe(0);
  });

  it("stages the mask's objects on the write box with the review's overlay, and accepts them into the box", async () => {
    const infer = fakeInfer(LEVEL);
    const planned = await planPlane(infer, dataset(LEVEL), volume(VOLUME_SHAPE), [201, 403, 606], 0, 16);
    const image = await readImage(infer, dataset(LEVEL), planned);
    // Two blocks of the 32 × 32 image: two objects of the 16 × 16 write box.
    const mask = Uint8Array.from({ length: 32 * 32 }, (_, i) => (Math.floor(i / 32) < 8 && i % 32 < 8) || (Math.floor(i / 32) >= 20 && i % 32 >= 20) ? 1 : 0);
    const staged = await stageMask(python!, planned.plan, mask, image, [255, 0, 0]);
    expect(staged.objects.length).toBe(16 * 16);
    expect(new Set(staged.objects)).toEqual(new Set([0, 1, 2]));
    expect(staged.maskFraction).toBeCloseTo((16 + 36) / 256);
    expect(staged.overlay.data[0]).toBeGreaterThan(image.data[0]);

    // In the write box (z, y, x), x mirrored back: the first block is rows
    // 0..3, x 12..15, the second rows 10..15, x 0..5. Accept into a uint8 box
    // holding 7 under the first: it keeps 7, the second gets the lowest free
    // id, the rest is background.
    const existing = new Uint8Array(256);
    existing[12] = 7;
    const args = { depth_axis: 0, shape: [1, 16, 16], dtype: "uint8", instance_target: false, overwrite: false, beside: [false, false] };
    const answer = await python!<{ value: Record<string, number>; arrays: ArrayBuffer[] }>(`${AI_PYTHON}:accept`, args, [staged.objects.slice().buffer, existing.slice().buffer]);
    expect(answer.value).toEqual({ n_foreground: 16 + 36 - 1, n_background: 256 - 16 - 36, n_overwritten: 0 });
    const after = new Uint8Array(answer.arrays[0]);
    expect([after[12], after[13], after[15 * 16], after[8 * 16]]).toEqual([7, 7, 2, 1]);

    // Every uint8 id is in use: an object on an unpainted voxel is refused, in the Python's words.
    const full = Uint8Array.from({ length: 256 }, (_, i) => (i < 254 ? i + 2 : 0));
    const lastVoxel = Int32Array.from({ length: 256 }, (_, i) => (i === 255 ? 1 : 0));
    const refused = await python!<{ refused?: string }>(`${AI_PYTHON}:accept`, args, [lastVoxel.buffer, full.buffer]);
    expect(refused.refused).toMatch(/do not fit in the volume's uint8 labels/);
  });
});
