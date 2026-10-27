// The training form's Channel
// (cellmap_flow/dashboard/static/js/dashboard/finetune/training-form.js),
// which both modes run, for a model of several channels that trains one at
// a time: training sends the channel picked as select_channel, and nothing
// until one is or for a model without a Channel; the pick is kept for its
// own model, also through the model picker's change, which saves the form
// while the Channel still holds the old model's channels; and Use this
// model's pick for a model not picked yet (pickChannel) is its Channel once
// it is, where Fit to Paint's (fillChannel) only fills a Channel nothing is
// picked in. A wrong channel here trains the wrong one without a word.
import { fileURLToPath } from "node:url";
import { afterEach, expect, it, vi } from "vitest";

const TRAINING_FORM = fileURLToPath(new URL("../../../cellmap_flow/dashboard/static/js/dashboard/finetune/training-form.js", import.meta.url));

interface TrainingForm {
  initTrainingForm: () => { save: () => void; trainingParams: () => Record<string, unknown> };
  setChannelChoice: (model: Record<string, unknown> | null) => void;
  pickChannel: (name: string, index: number) => void;
  fillChannel: (name: string, index: number) => boolean;
}

class FakeOption {
  constructor(public text: string, public value: string) {}
}

// The few DOM properties the form reads and writes, on plain objects: the
// unit tests run in node, which has no DOM. A select's value is "" unless
// one of its options has it, and its first option's once they are
// replaced, as in a browser.
class FakeElement {
  checked = false;
  disabled = false;
  hidden = false;
  type = "";
  textContent = "";
  className = "";
  title = "";
  dataset: Record<string, string> = {};
  style: Record<string, string> = {};
  options: FakeOption[] = [];
  private current = "";

  get value(): string {
    return this.current;
  }

  set value(value: string) {
    this.current = !this.options.length || this.options.some((o) => o.value === value) ? value : "";
  }

  replaceChildren(...options: FakeOption[]): void {
    this.options = options;
    this.current = options[0]?.value ?? "";
  }

  addEventListener(): void {}
}

async function setUp() {
  const elements = new Map<string, FakeElement>();
  const element = (id: string): FakeElement => {
    if (!elements.has(id)) elements.set(id, new FakeElement());
    return elements.get(id)!;
  };
  element("page-data").textContent = JSON.stringify({ mode: "server" });
  element("trainChannel").options = [new FakeOption("Pick the channel your paint is for", "")];
  const stored = new Map<string, string>();
  vi.stubGlobal("window", { localStorage: { getItem: (k: string) => stored.get(k) ?? null, setItem: (k: string, v: string) => stored.set(k, v) } });
  vi.stubGlobal("document", { getElementById: element, hidden: false, addEventListener: () => {}, removeEventListener: () => {} });
  vi.stubGlobal("Option", FakeOption);
  vi.stubGlobal("fetch", async () => ({ ok: true, status: 200, json: async () => ({}) }));
  const module = (await import(/* @vite-ignore */ TRAINING_FORM)) as TrainingForm;
  return { element, module, form: module.initTrainingForm() };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

it("sends the Channel picked for the model picked, kept per model", async () => {
  const { element, module, form } = await setUp();
  const run08 = { name: "run08", channel_choice: true, output_channels: 3, channels: ["all_mem", "mito", "er"] };
  const run07 = { name: "run07", channel_choice: true, output_channels: 2, channels: null };
  const channel = element("trainChannel");
  const names = () => channel.options.slice(1).map((option) => option.text);
  // As the model picker's change runs: the form saved, then the Channel
  // filled for the model (models.js displayModelInfo).
  const pick = (model: Record<string, unknown>) => {
    element("modelSelect").value = String(model.name);
    form.save();
    module.setChannelChoice(model);
  };

  pick(run08);
  expect(element("trainChannelGroup").hidden).toBe(false);
  expect(names()).toEqual(["all_mem", "mito", "er"]);
  expect(form.trainingParams()).not.toHaveProperty("select_channel");
  channel.value = "1";
  form.save();
  expect(form.trainingParams().select_channel).toBe(1);

  pick(run07);
  expect(names()).toEqual(["channel 0", "channel 1"]);
  expect(form.trainingParams()).not.toHaveProperty("select_channel");
  pick(run08);
  expect(form.trainingParams().select_channel).toBe(1);

  module.pickChannel("run07", 0);
  expect(form.trainingParams().select_channel).toBe(1);
  pick(run07);
  expect(form.trainingParams().select_channel).toBe(0);
  expect([module.fillChannel("run07", 1), module.fillChannel("run08", 2)]).toEqual([false, false]);
  expect(form.trainingParams().select_channel).toBe(0);
  channel.value = "";
  expect(module.fillChannel("run07", 1)).toBe(true);
  expect(form.trainingParams().select_channel).toBe(1);

  pick({ name: "affinities", channel_choice: false, output_channels: 3 });
  expect(element("trainChannelGroup").hidden).toBe(true);
  expect(form.trainingParams()).not.toHaveProperty("select_channel");
});
