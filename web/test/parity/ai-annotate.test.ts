// Parity of web/src/ai-annotate (geometry.ts, plane.ts, mask.ts, labels.ts,
// fill.ts, prompts.ts, organelles.ts) with cellmap_flow/ai_annotate
// (geometry.py, pipeline.py, mask_extraction.py, prompts.py, organelles.py)
// and cellmap_flow/finetune/session/fill.py.
//
// The fixture (test/fixtures/ai_annotate.json) was written by the Python
// functions themselves (test/fixtures/make_ai_fixtures.py, whose docstring
// lists the sections): plans for volumes, points, depth axes, crop sizes and
// data boxes, with their refusals' messages; raw boxes; masks found in an RGB
// image; planes normalized and resized; masks put on write boxes; ids for
// boxes with and without the planes beside; paint_box and restore_box on an
// 8^3 volume; the prompts word for word; the catalog. Integers and strings
// match exactly, floats within the parity tolerance. Regenerate the fixture
// when the Python changes; tests/web/test_fixtures_fresh.py fails when it is
// stale.
import { describe, expect, it } from "vitest";

import type { PaintVolume } from "../../src/contracts";
import { boxVoxels, chunkBytes, paintBox, readBox, restoreBox, volumeCornerNm, writeBox } from "../../src/ai-annotate/fill";
import { clickPx, fromScreen, planeAxes, planeName, planPlane, rawBox, screenShape, toScreen, writeShape } from "../../src/ai-annotate/geometry";
import { freshIds, seedLabels } from "../../src/ai-annotate/labels";
import { extractMask, normalizeToUint8 } from "../../src/ai-annotate/mask";
import { findOrganelleProfile, ORGANELLES, resolveOrganelleProfile } from "../../src/ai-annotate/organelles";
import { labelsForBox, maskToWriteShape, resizePlane } from "../../src/ai-annotate/plane";
import { buildRecolorPrompt, editablePrompt } from "../../src/ai-annotate/prompts";
import { AIAnnotateError, type DepthAxis, type Labels, MAX_IMAGE_SIDE, MAX_PROMPT_CHARS, type OrganelleProfile, PLANE_NAMES, type PlanePlan, type UndoEntry } from "../../src/ai-annotate/types";
import { CHUNK_DIR, chunkKey } from "../../src/finetune/volume";
import { MemoryPaintStore } from "../../src/state/paint-store";
import { type ArrayRecord, expectArrayMatches, loadFixture } from "./helpers";

interface Refusal { category: string; message: string; http_status: number }
interface VolumeRecord { output_voxel_size: number[]; input_voxel_size: number[]; dataset_offset_nm: number[] }
interface PlanCase {
  id: string; volume: VolumeRecord; volume_shape: number[]; point_nm: number[]; depth_axis: DepthAxis; crop_size_px: number;
  data_box_nm: [number[], number[]] | null;
  plan?: PlanePlan; click_px?: number[]; plane_axes?: number[]; write_shape?: number[]; plane_name?: string; error?: Refusal;
}
interface PaintCase {
  id: string; dtype: "uint8" | "uint16"; shape: number[]; chunks: number[]; initial: ArrayRecord; lo: number[]; hi: number[]; labels: ArrayRecord;
  overwrite: boolean; counts: { n_foreground: number; n_background: number; n_overwritten: number }; after: ArrayRecord;
  undo: { before: ArrayRecord; after: ArrayRecord } | null;
  restore?: { stroke: [number[], number][]; restored: number; final: ArrayRecord };
}
interface Fixture {
  seed: number;
  constants: { max_image_side: number; plane_names: string[] };
  screen: { depth_axis: DepthAxis; plane: ArrayRecord; to_screen: ArrayRecord; screen_shape: number[]; image: ArrayRecord; from_screen: ArrayRecord }[];
  plan_plane: PlanCase[];
  raw_box: { plan: string; voxel_size: number[]; corner: number[]; begin: number[]; count: number[] }[];
  extract_mask: { image: ArrayRecord; cases: { color_name: string; target_rgb: [number, number, number]; threshold: number; mask: ArrayRecord }[] };
  normalize_to_uint8: { id: string; plane: ArrayRecord; uint8: ArrayRecord }[];
  resize: { id: string; plane: ArrayRecord; shape: number[]; resized: ArrayRecord }[];
  mask_to_write_shape: { plan: string; mask: ArrayRecord; write: ArrayRecord }[];
  box_voxels: { id: string; volume: VolumeRecord; offset_nm: number[]; shape_nm: number[]; volume_shape: number[]; corner_nm: number[]; box: { lo: number[]; hi: number[] } | null }[];
  fresh_ids: { id: string; existing: ArrayRecord; count: number; count_up: boolean; ids?: number[]; error?: string }[];
  seed_labels: { id: string; objects: ArrayRecord; existing: ArrayRecord; count_up: boolean; labels?: ArrayRecord; error?: string }[];
  labels_for_box: { id: string; mask: ArrayRecord; existing: ArrayRecord; depth_axis: DepthAxis; count_up: boolean; beside: (ArrayRecord | null)[] | null; labels: ArrayRecord }[];
  paint_box: PaintCase[];
  prompts: {
    max_prompt_chars: number; editable: Record<string, string>; editable_generic: string;
    cases: { id: string; profile: OrganelleProfile; resolution_nm: number | null; prompt_override: string | null; prompt?: string; error?: Refusal }[];
  };
  organelles: { catalog: OrganelleProfile[]; lookups: { name: string; key: string | null }[]; resolved: { name: string; profile: OrganelleProfile }[] };
}

const fixture = loadFixture<Fixture>("ai_annotate.json");

/** A fixture array as the labels' typed array (uint8 or uint16, the paint layer's dtypes). */
function labelsOf(record: ArrayRecord): Labels {
  if (record.dtype === "uint16") return Uint16Array.from(record.data);
  if (record.dtype === "uint8") return Uint8Array.from(record.data);
  throw new Error(`${record.dtype} is not a labels dtype`);
}

const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

/** Expect a refusal with the Python's category, message and status. */
function expectRefused(fn: () => unknown, error: Refusal, label: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (e) {
    caught = e;
  }
  expect(caught, `${label}: raised`).toBeInstanceOf(AIAnnotateError);
  const e = caught as AIAnnotateError;
  expect(e.category, `${label}: category`).toBe(error.category);
  expect(e.httpStatus, `${label}: status`).toBe(error.http_status);
  expect(e.userMessage, `${label}: message`).toBe(error.message);
}

const planCases = new Map(fixture.plan_plane.map((c) => [c.id, c]));
const planOf = (id: string): PlanePlan => {
  const c = planCases.get(id);
  if (!c?.plan) throw new Error(`no plan ${id} in the fixture`);
  return c.plan;
};

describe("constants", () => {
  it("are the Python's", () => {
    expect(MAX_IMAGE_SIDE).toBe(fixture.constants.max_image_side);
    expect([...PLANE_NAMES]).toEqual(fixture.constants.plane_names);
    expect(MAX_PROMPT_CHARS).toBe(fixture.prompts.max_prompt_chars);
  });
});

describe("screen layout (to_screen, from_screen, screen_shape)", () => {
  for (const c of fixture.screen) {
    it(`depth axis ${c.depth_axis}`, () => {
      const [rows, cols] = c.plane.shape;
      const screen = toScreen(Int32Array.from(c.plane.data), rows, cols, c.depth_axis);
      expect([screen.rows, screen.cols]).toEqual(c.to_screen.shape);
      expect(Array.from(screen.data)).toEqual(c.to_screen.data);
      expect(screenShape([rows, cols], c.depth_axis)).toEqual(c.screen_shape);
      const [imageRows, imageCols] = c.image.shape;
      const back = fromScreen(Int32Array.from(c.image.data), imageRows, imageCols, c.depth_axis);
      expect([back.rows, back.cols]).toEqual(c.from_screen.shape);
      expect(Array.from(back.data)).toEqual(c.from_screen.data);
    });
  }
});

describe("plan_plane", () => {
  for (const c of fixture.plan_plane) {
    it(c.id, () => {
      const plan = () => planPlane(c.volume, c.volume_shape, c.point_nm, c.depth_axis, c.crop_size_px, c.data_box_nm);
      if (c.error) {
        expectRefused(plan, c.error, c.id);
        return;
      }
      const got = plan(), want = c.plan!;
      expect(got.depth_axis).toBe(want.depth_axis);
      expect(got.image_shape).toEqual(want.image_shape);
      expect(got.write_lo).toEqual(want.write_lo);
      expect(got.write_hi).toEqual(want.write_hi);
      for (const key of ["point_nm", "raw_offset_nm", "raw_shape_nm", "input_voxel_size"] as const) {
        expect(got[key].length, key).toBe(3);
        got[key].forEach((v, a) => expect(v, `${key}[${a}]`).toBeCloseTo(want[key][a], 9));
      }
      expect(got.resolution_nm).toBeCloseTo(want.resolution_nm, 9);
      expect(clickPx(got)).toEqual(c.click_px);
      expect(planeAxes(got)).toEqual(c.plane_axes);
      expect(writeShape(got)).toEqual(c.write_shape);
      expect(planeName(got)).toBe(c.plane_name);
    });
  }

  it("covers each depth axis, a refusal of each kind, a clipped field, anisotropic voxels and a half-pixel rounding", () => {
    const ids = new Set(planCases.keys());
    for (const id of ["centred_z", "centred_y", "centred_x", "clipped_edge", "aniso_xz", "data_box_clip", "half_pixel_rounding"]) expect(ids, id).toContain(id);
    expect(planCases.get("outside_before_z")!.error!.message).toContain("outside the annotation volume");
    expect(planCases.get("data_box_refused")!.error!.message).toContain("outside the data");
    // 1001 voxels at 16 nm pixels is 500.5 px: numpy rounds it to 500.
    expect(planOf("half_pixel_rounding").image_shape).toEqual([1024, 500]);
  });
});

describe("raw_box", () => {
  for (const c of fixture.raw_box) {
    it(`${c.plan} on a ${c.voxel_size.join("×")} nm grid from ${c.corner.join(", ")}`, () => {
      const box = rawBox(planOf(c.plan), c.voxel_size, c.corner);
      expect(box.begin).toEqual(c.begin);
      expect(box.count).toEqual(c.count);
    });
  }
});

describe("extract_mask", () => {
  const { image } = fixture.extract_mask;
  const [height, width] = image.shape;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data.set(image.data.slice(i * 3, i * 3 + 3), i * 4);
    data[i * 4 + 3] = 255;
  }
  const rgba = { width, height, data };
  for (const c of fixture.extract_mask.cases) {
    it(`${c.color_name} (${c.target_rgb.join(", ")}) at ${c.threshold}`, () => {
      expectArrayMatches(extractMask(rgba, c.target_rgb, c.threshold), c.mask, c.color_name);
    });
  }
});

describe("normalize_to_uint8", () => {
  for (const c of fixture.normalize_to_uint8) {
    it(c.id, () => {
      const plane = c.plane.dtype === "float32" ? Float32Array.from(c.plane.data) : c.plane.dtype === "uint16" ? Uint16Array.from(c.plane.data) : Uint8Array.from(c.plane.data);
      expectArrayMatches(normalizeToUint8(plane, c.plane.dtype === "uint8"), c.uint8, c.id);
    });
  }
});

describe("_resize (PIL BOX when shrinking, BILINEAR otherwise)", () => {
  for (const c of fixture.resize) {
    it(c.id, () => {
      const [rows, cols] = c.plane.shape;
      const out = resizePlane({ rows, cols, data: Float32Array.from(c.plane.data) }, c.shape[0], c.shape[1]);
      expect([out.rows, out.cols]).toEqual(c.shape);
      expect(c.resized.dtype).toBe("float32");
      expectArrayMatches(out.data, c.resized, c.id);
    });
  }
});

describe("mask_to_write_shape", () => {
  for (const c of fixture.mask_to_write_shape) {
    it(c.plan, () => {
      const plan = planOf(c.plan);
      expect(c.mask.shape).toEqual(plan.image_shape);
      const out = maskToWriteShape(Uint8Array.from(c.mask.data), plan);
      expect(c.write.shape).toEqual(writeShape(plan));
      expectArrayMatches(out, c.write, c.plan);
    });
  }
});

describe("box_voxels and volume_corner_nm", () => {
  for (const c of fixture.box_voxels) {
    it(c.id, () => {
      volumeCornerNm(c.volume.dataset_offset_nm, c.volume.output_voxel_size).forEach((v, a) => expect(v).toBeCloseTo(c.corner_nm[a], 9));
      expect(boxVoxels(c.volume, c.offset_nm, c.shape_nm, c.volume_shape)).toEqual(c.box);
    });
  }
});

describe("_fresh_ids", () => {
  for (const c of fixture.fresh_ids) {
    it(c.id, () => {
      const existing = labelsOf(c.existing);
      if (c.error !== undefined) {
        expectRefused(() => freshIds(existing, c.count, c.count_up), { category: "refused", message: c.error, http_status: 409 }, c.id);
        return;
      }
      expect(freshIds(existing, c.count, c.count_up)).toEqual(c.ids);
    });
  }
});

describe("seed_labels", () => {
  for (const c of fixture.seed_labels) {
    it(c.id, () => {
      const existing = labelsOf(c.existing);
      const objects = Int32Array.from(c.objects.data);
      if (c.error !== undefined) {
        expectRefused(() => seedLabels(objects, existing, c.count_up), { category: "refused", message: c.error, http_status: 409 }, c.id);
        return;
      }
      const labels = seedLabels(objects, existing, c.count_up);
      expect(labels.constructor).toBe(existing.constructor);
      expectArrayMatches(labels, c.labels!, c.id);
    });
  }
});

describe("labels_for_box", () => {
  for (const c of fixture.labels_for_box) {
    it(c.id, () => {
      // The box is one voxel thick along the depth axis, so its C-order
      // layout is the plane's: writeLo 0, writeHi the box's shape.
      const existing = labelsOf(c.existing);
      const beside = (c.beside ?? []).map((p) => (p ? labelsOf(p) : null));
      const labels = labelsForBox(Uint8Array.from(c.mask.data), existing, [0, 0, 0], c.existing.shape, c.depth_axis, c.count_up, beside);
      expect(labels.constructor).toBe(existing.constructor);
      expectArrayMatches(labels, c.labels, c.id);
    });
  }
});

describe("paint_box and restore_box", () => {
  /** A paint volume of the case's shape and chunks in a fresh store, its
   * nonzero chunks written as the brush would have written them (a chunk
   * never painted has no file, and reads as zeros). */
  async function storeOf(c: PaintCase): Promise<{ store: MemoryPaintStore; volume: PaintVolume }> {
    const store = new MemoryPaintStore();
    const volume: PaintVolume = {
      id: "vol-1", type: "annotation_volume", model_name: "m", model_id: "m", dataset_path: "x", chunk_size: c.chunks, input_size: c.chunks,
      output_voxel_size: [8, 8, 8], input_voxel_size: [8, 8, 8], dataset_offset_nm: [4, 4, 4], dataset_shape_voxels: c.shape, dtype: c.dtype,
      input_norm: [], postprocess: [], created_at: "2026-10-15T12:00:00Z",
    };
    const initial = labelsOf(c.initial);
    const [nz, ny, nx] = c.shape, [cz, cy, cx] = c.chunks;
    for (let iz = 0; iz < nz / cz; iz++) for (let iy = 0; iy < ny / cy; iy++) for (let ix = 0; ix < nx / cx; ix++) {
      const chunk = c.dtype === "uint16" ? new Uint16Array(cz * cy * cx) : new Uint8Array(cz * cy * cx);
      let any = false;
      for (let z = 0; z < cz; z++) for (let y = 0; y < cy; y++) for (let x = 0; x < cx; x++) {
        const v = initial[((iz * cz + z) * ny + iy * cy + y) * nx + ix * cx + x];
        chunk[(z * cy + y) * cx + x] = v;
        if (v) any = true;
      }
      if (any) await store.put(volume.id, CHUNK_DIR + chunkKey([iz, iy, ix]), chunkBytes(chunk));
    }
    return { store, volume };
  }

  for (const c of fixture.paint_box) {
    it(c.id, async () => {
      const { store, volume } = await storeOf(c);
      const labels = labelsOf(c.labels);
      const undo: UndoEntry[] = [];
      const counts = await paintBox(store, volume, c.lo, c.hi, () => labels.slice(), { overwrite: c.overwrite, undo });
      expect(counts).toEqual(c.counts);
      const whole = [0, 0, 0];
      expectArrayMatches(await readBox(store, volume, whole, c.shape), c.after, `${c.id}: after`);
      if (!c.undo) {
        expect(undo).toEqual([]);
        // Nothing written: no chunk file appeared.
        const files = await store.list(volume.id, CHUNK_DIR);
        const initial = labelsOf(c.initial);
        const { store: fresh } = await storeOf(c);
        expect(files).toEqual(await fresh.list(volume.id, CHUNK_DIR));
        expectArrayMatches(await readBox(store, volume, whole, c.shape), c.initial, `${c.id}: unchanged`);
        expect(initial.length).toBe(prod(c.shape));
      } else {
        expect(undo.length).toBe(1);
        expect(undo[0].lo).toEqual(c.lo);
        expect(undo[0].hi).toEqual(c.hi);
        expectArrayMatches(undo[0].before, c.undo.before, `${c.id}: undo before`);
        expectArrayMatches(undo[0].after, c.undo.after, `${c.id}: undo after`);
      }
      if (c.restore) {
        for (const [index, value] of c.restore.stroke) {
          const one = c.dtype === "uint16" ? Uint16Array.of(value) : Uint8Array.of(value);
          await writeBox(store, volume, index, index.map((i) => i + 1), one);
        }
        expect(await restoreBox(store, volume, undo[0])).toBe(c.restore.restored);
        expectArrayMatches(await readBox(store, volume, whole, c.shape), c.restore.final, `${c.id}: restored`);
      }
    });
  }
});

describe("prompts", () => {
  it("editable_prompt, per catalog entry and for a generic profile", () => {
    for (const [key, text] of Object.entries(fixture.prompts.editable)) expect(editablePrompt(ORGANELLES[key]), key).toBe(text);
    expect(editablePrompt(resolveOrganelleProfile("golgi"))).toBe(fixture.prompts.editable_generic);
  });
  for (const c of fixture.prompts.cases) {
    it(`build_recolor_prompt: ${c.id}`, () => {
      const build = () => buildRecolorPrompt(c.profile, c.resolution_nm, c.prompt_override);
      if (c.error) expectRefused(build, c.error, c.id);
      else expect(build()).toBe(c.prompt);
    });
  }
});

describe("organelles", () => {
  it("the catalog, verbatim and in order", () => {
    expect(Object.values(ORGANELLES)).toEqual(fixture.organelles.catalog);
  });
  for (const l of fixture.organelles.lookups) {
    it(`find_organelle_profile(${JSON.stringify(l.name)})`, () => {
      expect(findOrganelleProfile(l.name)?.key ?? null).toBe(l.key);
    });
  }
  for (const r of fixture.organelles.resolved) {
    it(`resolve_organelle_profile(${JSON.stringify(r.name)})`, () => {
      expect(resolveOrganelleProfile(r.name)).toEqual(r.profile);
    });
  }
});
