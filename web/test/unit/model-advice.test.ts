// The dashboard's model-advice banner
// (cellmap_flow/dashboard/static/js/dashboard/model-advice.js), which both
// modes run: a suggestion turned down stays down when the chain changes, as
// it does after every Apply, and advice about something new still shows.
import { fileURLToPath } from "node:url";
import { afterAll, expect, it, vi } from "vitest";

// The few DOM calls the banner makes, on a tree of plain objects: the unit
// tests run in node, which has no DOM.
class FakeElement {
  children: FakeElement[] = [];
  parent: FakeElement | null = null;
  className = "";
  innerHTML = "";
  textContent = "";
  onclick: (() => void) | null = null;

  appendChild(child: FakeElement): FakeElement {
    child.remove();
    child.parent = this;
    this.children.push(child);
    return child;
  }

  remove(): void {
    if (!this.parent) return;
    this.parent.children = this.parent.children.filter((c) => c !== this);
    this.parent = null;
  }
}

const MODEL_ADVICE = fileURLToPath(new URL("../../../cellmap_flow/dashboard/static/js/dashboard/model-advice.js", import.meta.url));

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
      input_norm: { MinMaxNormalizer: { name: "MinMaxNormalizer", min_value: 0, max_value: 255, invert: false } },
      order: ["MinMaxNormalizer"],
      reason: "uint8 raw",
      confidence: "high",
    },
    postprocess_review: { level: "suggest", message: "The output is unbounded.", suggest: ["SigmoidPostprocessor"], params: {} },
    ...changes,
  };
}

afterAll(() => {
  vi.unstubAllGlobals();
});

it("keeps a dismissed suggestion hidden when only the chain changed", async () => {
  const banner = new FakeElement();
  let models: Record<string, unknown>[] = [];
  vi.stubGlobal("document", {
    getElementById: () => banner,
    createElement: () => new FakeElement(),
    addEventListener: () => {},
  });
  vi.stubGlobal("fetch", async () => ({ ok: true, status: 200, json: async () => ({ success: true, models }) }));
  const advice = (await import(/* @vite-ignore */ MODEL_ADVICE)) as {
    initModelAdvice: (chains: unknown) => void;
    refreshModelAdvice: () => void;
  };
  advice.initModelAdvice({ input: {}, postprocess: {} });
  // One poll of the advice, answered with `answer`; every probe has
  // answered, so the poll stops after it.
  async function poll(answer: Record<string, unknown>[]): Promise<number> {
    models = answer;
    advice.refreshModelAdvice();
    await new Promise((resolve) => setTimeout(resolve, 0));
    return banner.children.length;
  }

  expect(await poll([advised()])).toBe(2);
  for (const suggestion of [...banner.children]) {
    suggestion.children.find((child) => child.className.includes("btn-link"))!.onclick!();
  }
  expect(banner.children).toHaveLength(0);

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
