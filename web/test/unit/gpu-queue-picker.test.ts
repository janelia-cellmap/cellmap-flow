// The GPU queue picker every chooser of GPU queues on the dashboard is
// (cellmap_flow/dashboard/static/js/shared/gpu-queues.js): a dropdown of
// checkboxes, every queue ticked until a choice is made, each item saying
// what LSF says of its queue, the last one ticked kept ticked, a queue LSF
// does not list kept rather than dropped, and the choice surviving each
// minute's answer. What it ticks is where the jobs go, so a box that drops
// out sends them somewhere else without a word.
import { fileURLToPath } from "node:url";
import { afterEach, expect, it, vi } from "vitest";

import { FakeElement } from "./dashboard-dom";

const GPU_QUEUES = fileURLToPath(new URL("../../../cellmap_flow/dashboard/static/js/shared/gpu-queues.js", import.meta.url));

interface Picker {
  value: () => string[];
  setValue: (queues: string[] | null) => void;
  render: (answer: unknown) => void;
}

interface GpuQueues {
  createQueuePicker: (root: unknown, options?: { onChange?: (queues: string[]) => void }) => Picker;
}

const ANSWER = {
  available: true,
  queues: [
    { queue: "gpu_h100", label: "H100", open: true, description: "0/80 GPUs free, 496 queued" },
    { queue: "gpu_h200", label: "H200", open: true, description: "8/64 GPUs free, 0 queued" },
    { queue: "gpu_a100", label: "A100", open: true, description: "9/96 GPUs free, 2 queued" },
    { queue: "gpu_rtx6000", label: "RTX 6000", open: false, description: "closed (Closed:Adm)" },
  ],
};

async function setUp() {
  vi.stubGlobal("document", { createElement: (tag: string) => new FakeElement(tag) });
  const { createQueuePicker } = (await import(/* @vite-ignore */ GPU_QUEUES)) as GpuQueues;
  const root = new FakeElement("div");
  const changes: string[][] = [];
  const picker = createQueuePicker(root, { onChange: (queues) => changes.push(queues) });
  const button = () => root.children[0];
  const boxes = () => root.querySelectorAll("input");
  // What each item says, and whether it is ticked.
  const items = () => boxes().map((box) => [box.value, box.checked]);
  const words = () => root.querySelectorAll("label span").map((span) => span.textContent);
  // A click on a box: it toggles, and its change bubbles to the list.
  const click = (queue: string) => {
    const box = boxes().find((b) => b.value === queue)!;
    box.checked = !box.checked;
    box.dispatchEvent({ type: "change" });
  };
  return { root, picker, changes, button, items, words, click };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

it("ticks every queue until a choice is made, and keeps one ticked", async () => {
  const { root, picker, changes, button, items, words, click } = await setUp();
  expect(picker.value()).toEqual([]);
  expect(root.children).toEqual([]);

  picker.render(ANSWER);
  expect(picker.value()).toEqual(["gpu_h100", "gpu_h200", "gpu_a100", "gpu_rtx6000"]);
  expect(button().textContent).toBe("H100, H200, A100, RTX 6000");
  expect(words()).toEqual([
    "H100 — 0/80 GPUs free, 496 queued", "H200 — 8/64 GPUs free, 0 queued",
    "A100 — 9/96 GPUs free, 2 queued", "RTX 6000 — closed (Closed:Adm)",
  ]);

  click("gpu_h100");
  click("gpu_rtx6000");
  expect(changes).toEqual([["gpu_h200", "gpu_a100", "gpu_rtx6000"], ["gpu_h200", "gpu_a100"]]);
  expect(button().textContent).toBe("H200, A100");
  click("gpu_h200");
  click("gpu_a100");
  expect(picker.value()).toEqual(["gpu_a100"]);
  expect(items().filter(([, checked]) => checked)).toEqual([["gpu_a100", true]]);
  expect(changes).toHaveLength(3);

  // Each minute's answer keeps the choice, and one with no queues changes nothing.
  picker.render(ANSWER);
  picker.render({ available: false, queues: [] });
  expect(items()).toEqual([["gpu_h100", false], ["gpu_h200", false], ["gpu_a100", true], ["gpu_rtx6000", false]]);
});

it("keeps a queue chosen that LSF does not list", async () => {
  const { picker, items, words } = await setUp();
  // The saved settings, read before LSF has answered: a YAML can name any queue.
  picker.setValue(["gpu_h200", "gpu_l4"]);
  expect(picker.value()).toEqual(["gpu_h200", "gpu_l4"]);
  picker.render(ANSWER);
  expect(items()).toEqual([
    ["gpu_h100", false], ["gpu_h200", true], ["gpu_a100", false], ["gpu_rtx6000", false], ["gpu_l4", true],
  ]);
  expect(words().at(-1)).toBe("gpu_l4 (current)");
  // No choice (a form saved before LSF answered) is every queue again.
  picker.setValue([]);
  expect(picker.value()).toEqual(["gpu_h100", "gpu_h200", "gpu_a100", "gpu_rtx6000"]);
});
