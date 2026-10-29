// What a tab lists of what this browser keeps (src/kept): the finetuned
// models and paint volumes of earlier tabs and visits are in the stores but
// in none of a new tab's lists (the Hugging Face list, the ranking's
// candidates, the sessions to resume) until one is brought in; a reload of
// the tab lists what it listed; and Delete all takes the earlier ones out of
// the stores, a volume's superseded copies with it, leaving the tab's own.
import { afterEach, expect, it, vi } from "vitest";

import { createBackend } from "../../src/backend";
import type { ApiResponse, FinetunedRecord, PaintVolume } from "../../src/contracts";
import { createVolume, supersedeVolume } from "../../src/finetune/volume";
import { deleteEarlier, earlierKept } from "../../src/kept";
import { finetunedRecord } from "../../src/models/finetuned";
import { defaultChain, keepInTab, openSession } from "../../src/state";
import { DATASET, entry, type FakeContext, fakeContext, NAME_54, request } from "./backend-fakes";

function body(res: ApiResponse): any {
  if (!("json" in res)) throw new Error(`not a JSON answer: ${JSON.stringify(res)}`);
  return res.json;
}

function volume(id: string): PaintVolume {
  return {
    id, type: "annotation_volume", model_name: NAME_54, model_id: "cellmap/onnx-web-models/setup_54", dataset_path: DATASET.url,
    chunk_size: [2, 4, 4], input_size: [10, 12, 12], output_voxel_size: [8, 8, 8], input_voxel_size: [8, 8, 8],
    dataset_offset_nm: [4, 4, 4], dataset_shape_voxels: [4, 8, 8], dtype: "uint8",
    input_norm: defaultChain().input_norm, postprocess: defaultChain().postprocess, created_at: "2026-10-20T12:00:00.000Z",
  };
}

/** A model finetuned from setup_54 as `name`, as a run keeps it. */
function finetuned(name: string): FinetunedRecord {
  return finetunedRecord(entry("setup_54"), name, 16, { volume: "", created_at: "2026-10-20T12:00:00Z", params: {}, epochs: 1, final_loss: null });
}

afterEach(() => vi.unstubAllGlobals());

it("lists none of what earlier tabs kept in a new tab, one once it is brought in, and after a reload what the tab listed", async () => {
  const items = new Map<string, string>();
  vi.stubGlobal("sessionStorage", {
    getItem: (k: string) => items.get(k) ?? null,
    setItem: (k: string, v: string) => void items.set(k, v),
    removeItem: (k: string) => void items.delete(k),
  });
  const ctx: FakeContext = fakeContext();
  ctx.session = await openSession();  // a new tab
  const old = finetuned(`${NAME_54}-ft1`);
  await ctx.catalog.addFinetuned(old, new ArrayBuffer(16));
  const painted = volume("vol-00000001-20261020-120000");
  await createVolume(ctx.paint, painted);
  const backend = createBackend(ctx);
  const lists = async () => ({
    hf: Object.keys(body(await backend.handle(request("GET", "/api/huggingface-models")))),
    ranked: body(await backend.handle(request("GET", "/api/finetune/rank/candidates"))).candidates.map((c: { name: string }) => c.name),
    sessions: body(await backend.handle(request("POST", "/api/finetune/list-existing-sessions", {}))).sessions.map((s: { session_id: string }) => s.session_id),
  });

  const fresh = await lists();
  expect(fresh.hf).not.toContain(old.id);
  expect(fresh.ranked).not.toContain(old.name);
  expect(fresh.sessions).toEqual([]);
  // Kept in this browser lists them instead.
  expect(await earlierKept(ctx)).toMatchObject({ finetuned: [{ id: old.id }], volumes: [{ volume: { id: painted.id }, ids: [painted.id] }] });

  // Use in this tab, for each.
  await keepInTab(ctx.session, { finetuned: [old.id], volumes: [painted.id] });
  const adopted = await lists();
  expect(adopted.hf).toContain(old.id);
  expect(adopted.ranked).toContain(old.name);
  expect(adopted.sessions).toEqual([painted.id]);
  expect(await earlierKept(ctx)).toEqual({ finetuned: [], volumes: [] });

  ctx.session = await openSession();  // the same tab, reloaded
  expect(await lists()).toEqual(adopted);
});

it("deletes every earlier finetuned model and paint volume on Delete all, a volume's superseded copy with it, and keeps the tab's", async () => {
  const ctx = fakeContext();
  const old = finetuned(`${NAME_54}-ft1`), mine = finetuned(`${NAME_54}-ft2`);
  for (const record of [old, mine]) await ctx.catalog.addFinetuned(record, new ArrayBuffer(16));
  const [first, carried, ours] = ["vol-00000001-20261020-120000", "vol-00000002-20261020-130000", "vol-00000003-20261021-090000"].map(volume);
  for (const v of [first, carried, ours]) await createVolume(ctx.paint, v);
  await supersedeVolume(ctx.paint, first.id, carried.id);  // an earlier tab's Use this model carried its paint
  await keepInTab(ctx.session, { finetuned: [mine.id] });  // finetuned here
  await ctx.session.update({ paintVolume: ours.id });  // painted here
  expect((await earlierKept(ctx)).volumes.map((g) => g.ids.sort())).toEqual([[first.id, carried.id]]);

  await deleteEarlier(ctx);
  expect((await ctx.catalog.finetuned()).map((r) => r.id)).toEqual([mine.id]);
  expect(await ctx.paint.volumes()).toEqual([ours.id]);
  expect(await earlierKept(ctx)).toEqual({ finetuned: [], volumes: [] });
});
