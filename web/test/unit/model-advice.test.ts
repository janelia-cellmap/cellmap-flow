// The dashboard's model advice
// (cellmap_flow/dashboard/static/js/dashboard/model-advice.js), which both
// modes run: a suggestion turned down stays down when the chain changes, as
// it does after every Apply, and advice about something new still shows; a
// suggestion's Apply applies the chain, through the Apply bar's apply; and
// the advice is asked for until every model's probe has answered, and again
// when the backend changes the chain itself.
import { fileURLToPath } from "node:url";
import { afterEach, expect, it, vi } from "vitest";

import { FakeElement } from "./dashboard-dom";

const MODEL_ADVICE = fileURLToPath(new URL("../../../cellmap_flow/dashboard/static/js/dashboard/model-advice.js", import.meta.url));
const EVENTS = fileURLToPath(new URL("../../../cellmap_flow/dashboard/static/js/lib/events.js", import.meta.url));

interface Advice {
  initModelAdvice: (processing: unknown) => void;
  refreshModelAdvice: () => void;
}

// The page as the advice sees it: one element for both of its boxes (the
// banner above the tabs, the Processing tab's), and GET /api/model_advice
// answering `models`, or failing while `failing`. The module is imported
// once for the file, so each test names its own models: what one test
// dismissed stays dismissed in the next.
async function setUp(processing: Record<string, unknown> = { input: {}, postprocess: {}, apply: () => {} }) {
  const box = new FakeElement("div");
  const page = { models: [] as Record<string, unknown>[], failing: false, requests: 0 };
  // The page's events (lib/events.js announces them on document).
  const events = new EventTarget();
  vi.stubGlobal("document", {
    getElementById: () => box,
    createElement: (tag: string) => new FakeElement(tag),
    addEventListener: events.addEventListener.bind(events),
    removeEventListener: events.removeEventListener.bind(events),
    dispatchEvent: events.dispatchEvent.bind(events),
  });
  vi.stubGlobal("fetch", async () => {
    page.requests += 1;
    if (page.failing) throw new TypeError("Failed to fetch");
    return { ok: true, status: 200, json: async () => ({ success: true, models: page.models }) };
  });
  const advice = (await import(/* @vite-ignore */ MODEL_ADVICE)) as Advice;
  advice.initModelAdvice(processing);
  return { box, page, advice };
}

// The input normalization suggested for uint8 raw.
const UINT8 = { MinMaxNormalizer: { name: "MinMaxNormalizer", min_value: 0, max_value: 255, invert: false } };

// GET /api/model_advice's entry for a model whose probe has answered, with
// a suggestion on each side; `changes` replaces fields of it.
function advised(changes: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    model: "mito",
    probe_available: true,
    configured_input_norm: [],
    configured_postprocess: [],
    input_norm_matches: false,
    input_norm_differences: [],
    input_norm_suggestion: {
      input_norm: UINT8,
      order: ["MinMaxNormalizer"],
      reason: "uint8 raw",
      confidence: "high",
    },
    postprocess_review: { level: "suggest", message: "The output is unbounded.", suggest: ["SigmoidPostprocessor"], params: {} },
    ...changes,
  };
}

function button(suggestion: FakeElement, className: string): FakeElement {
  return suggestion.children.find((child) => child.className.includes(className))!;
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it("keeps a dismissed suggestion hidden when only the chain changed", async () => {
  const { box, page, advice } = await setUp();
  // One poll of the advice, answered with `answer`; every probe has
  // answered, so the poll stops after it.
  async function poll(answer: Record<string, unknown>[]): Promise<number> {
    page.models = answer;
    advice.refreshModelAdvice();
    await new Promise((resolve) => setTimeout(resolve, 0));
    return box.children.length;
  }

  expect(await poll([advised()])).toBe(2);
  for (const suggestion of [...box.children]) button(suggestion, "btn-link").onclick!();
  expect(box.children).toHaveLength(0);

  // After an Apply of other steps the server says the same, about another chain.
  const applied = { configured_input_norm: ["LambdaNormalizer"], configured_postprocess: ["ThresholdPostprocessor"] };
  expect(await poll([advised(applied)])).toBe(0);

  // Other values to suggest for the model are new advice.
  const otherRange = {
    ...applied,
    input_norm_suggestion: {
      input_norm: { MinMaxNormalizer: { name: "MinMaxNormalizer", min_value: 0, max_value: 65535, invert: false } },
      order: ["MinMaxNormalizer"],
      reason: "uint16 raw",
      confidence: "high",
    },
  };
  expect(await poll([advised(otherRange)])).toBe(1);
});

it("applies a suggestion in one click, through the Apply bar's apply", async () => {
  const processing = { input: { apply: vi.fn() }, postprocess: { apply: vi.fn() }, apply: vi.fn() };
  const { box, page, advice } = await setUp(processing);
  page.models = [advised({ model: "applied" })];
  advice.refreshModelAdvice();
  await new Promise((resolve) => setTimeout(resolve, 0));
  const [inputSuggestion, postSuggestion] = box.children;

  button(inputSuggestion, "btn-primary").onclick!();
  expect(processing.input.apply).toHaveBeenCalledWith(["MinMaxNormalizer"], UINT8);
  expect(processing.apply).toHaveBeenCalledTimes(1);
  button(postSuggestion, "btn-primary").onclick!();
  expect(processing.postprocess.apply).toHaveBeenCalledWith(["SigmoidPostprocessor"], {});
  expect(processing.apply).toHaveBeenCalledTimes(2);
  // Applied, so gone.
  expect(box.children).toHaveLength(0);
});

it("asks for the advice until every model's probe has answered", async () => {
  vi.useFakeTimers();
  const { page, advice } = await setUp();
  // A job queued for ten minutes, its server unreachable for one of them.
  page.models = [advised({ model: "queued", probe_available: false })];
  advice.refreshModelAdvice();
  await vi.advanceTimersByTimeAsync(4 * 60_000);
  page.failing = true;
  await vi.advanceTimersByTimeAsync(60_000);
  page.failing = false;
  await vi.advanceTimersByTimeAsync(4 * 60_000);
  const asked = page.requests;
  await vi.advanceTimersByTimeAsync(60_000);
  expect(page.requests).toBeGreaterThan(asked);

  // Its probe answers: the advice is complete, and no more is asked.
  page.models = [advised({ model: "queued" })];
  await vi.advanceTimersByTimeAsync(60_000);
  const answered = page.requests;
  await vi.advanceTimersByTimeAsync(10 * 60_000);
  expect(page.requests).toBe(answered);
});

it("asks again when the backend changes the chain itself, as Fit to Paint does", async () => {
  // Every probe has answered, so the poll has stopped; Fit to Paint then
  // sets an affinity model's chain, whose chain into a segmentation is now
  // offered cut at the threshold the ranking fitted.
  vi.useFakeTimers();
  const { page, advice } = await setUp();
  const { announce, CHAIN_CHANGED_EVENT } = await import(/* @vite-ignore */ EVENTS);
  page.models = [advised({ model: "fitted" })];
  advice.refreshModelAdvice();
  await vi.advanceTimersByTimeAsync(10 * 60_000);
  const asked = page.requests;
  announce(CHAIN_CHANGED_EVENT);
  await vi.advanceTimersByTimeAsync(0);
  expect(page.requests).toBe(asked + 1);
});
