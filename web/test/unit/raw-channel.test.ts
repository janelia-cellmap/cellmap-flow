// The Processing tab's Channel (cellmap_flow/dashboard/static/js/dashboard/
// raw-channel.js), which both modes run: shown for a raw of several channels,
// named as GET /api/raw/channels names them; a pick makes the input list's
// ChannelSelector its first step and applies the chain; and the select
// follows the list.
import { fileURLToPath } from "node:url";
import { afterAll, expect, it, vi } from "vitest";

import { FakeElement, FakeNode } from "./dashboard-dom";

const RAW_CHANNEL = fileURLToPath(new URL("../../../cellmap_flow/dashboard/static/js/dashboard/raw-channel.js", import.meta.url));

type Step = { name: string } & Record<string, unknown>;

// The input list as raw-channel.js drives it (shared/op-chain.js): its chain,
// show() making the chain exactly some steps, and a listener after each edit.
function inputList(steps: Step[]) {
  const listeners: (() => void)[] = [];
  const list = {
    chain: steps,
    getChain: () => list.chain.map((step) => ({ ...step })),
    show(shown: Step[]) {
      list.chain = shown;
      listeners.forEach((listener) => listener());
    },
    onEdit: (listener: () => void) => listeners.push(listener),
  };
  return list;
}

// The Channel's row and select on a page whose backend answers `channels`,
// over an input list holding `steps`.
async function mount(channels: Record<string, unknown>, steps: Step[]) {
  const row = new FakeElement("div");
  row.hidden = true;
  const select = new FakeElement("select");
  const byId: Record<string, FakeElement> = { rawChannelRow: row, rawChannel: select };
  vi.stubGlobal("document", { getElementById: (id: string) => byId[id] ?? null, createElement: (tag: string) => new FakeElement(tag) });
  vi.stubGlobal("Node", FakeNode);
  vi.stubGlobal("fetch", async () => ({ ok: true, status: 200, json: async () => ({ success: true, ...channels }) }));
  const { initRawChannel } = (await import(/* @vite-ignore */ RAW_CHANNEL)) as {
    initRawChannel: (parts: { input: unknown; apply: () => void }) => Promise<void>;
  };
  const list = inputList(steps);
  const apply = vi.fn();
  await initRawChannel({ input: list, apply });
  return { row, select, list, apply };
}

afterAll(() => vi.unstubAllGlobals());

const MIN_MAX = { name: "MinMaxNormalizer", min_value: "0", max_value: "255" };

it("picks the channel every model reads as the input list's first step, and follows the list", async () => {
  const { row, select, list, apply } = await mount({ channels: 2, labels: ["DAPI", "GFP"], selected: 0 }, [MIN_MAX]);
  expect(row.hidden).toBe(false);
  expect(select.children.map((option) => [option.value, option.textContent])).toEqual([["0", "DAPI"], ["1", "GFP"]]);
  expect(select.value).toBe("0");

  select.value = "1";
  select.dispatchEvent({ type: "change" });
  expect(list.chain).toEqual([{ name: "ChannelSelector", channel: "1" }, MIN_MAX]);
  expect(apply).toHaveBeenCalledTimes(1);

  // The list's ChannelSelector unticked by hand: channel 0, as no step reads.
  list.show([MIN_MAX]);
  expect(select.value).toBe("0");
});

it("is hidden for a raw of one channel", async () => {
  const { row } = await mount({ channels: 1, labels: null, selected: 0 }, []);
  expect(row.hidden).toBe(true);
});
