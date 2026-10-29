// The offer of a new annotation volume, when an action is refused for want of
// one (cellmap_flow/dashboard/static/js/dashboard/finetune/requests.js
// offerNewVolume): once New Volume has made the volume, the action runs again,
// so its "no annotation volume" gives way to what it does; a New Volume that
// made none leaves nothing to run later. Both modes run this script.
import { afterEach, expect, it, vi } from "vitest";

// @ts-expect-error: the dashboard's scripts are plain JavaScript
import { VOLUME_CREATED, offerNewVolume } from "../../../cellmap_flow/dashboard/static/js/dashboard/finetune/requests.js";

afterEach(() => vi.unstubAllGlobals());

function page() {
  const doc = Object.assign(new EventTarget(), {
    getElementById: () => ({ scrollIntoView() {}, click() { doc.clicks += 1; } }),
    clicks: 0,
  });
  vi.stubGlobal("document", doc);
  vi.stubGlobal("confirm", () => true);
  return { doc, made: (ok: boolean) => doc.dispatchEvent(new CustomEvent(VOLUME_CREATED, { detail: { ok } })) };
}

it("runs the refused action again once New Volume has made the volume", () => {
  const { doc, made } = page();
  const again = vi.fn();

  expect(offerNewVolume("Could not start AI annotation: No annotation volume to annotate.", again)).toBe(true);
  expect(doc.clicks).toBe(1);
  expect(again).not.toHaveBeenCalled();
  made(true);
  made(true);

  expect(again).toHaveBeenCalledTimes(1);
});

it("leaves nothing to run when New Volume made no volume", () => {
  const { made } = page();
  const again = vi.fn();

  offerNewVolume("Could not mark region as good: No annotation volume.", again);
  made(false);
  made(true); // a later New Volume, clicked by hand

  expect(again).not.toHaveBeenCalled();
});
