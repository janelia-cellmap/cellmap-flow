// The training form's Channel
// (cellmap_flow/dashboard/static/js/dashboard/finetune/training-form.js),
// which both modes run, for a model of several channels that trains one at
// a time: training sends the channel picked as select_channel, and nothing
// until one is or for a model without a Channel; the pick is kept for its
// own model, also through the model picker's change, which saves the form
// while the Channel still holds the old model's channels; and Use this
// model's pick for a model not picked yet (pickChannel) is its Channel once
// it is, where Fit to Paint's (fillChannel) only fills a Channel nothing is
// picked in. A wrong channel here trains the wrong one without a word. And
// its Learning Rate, typed rather than picked from presets: sent as typed,
// put back as typed when the page loads again, and refused, saying why,
// when it is not a number above 0; its Ease down box, sent as the schedule
// (cosine ticked, constant not) and put back as it was left. And Restart's
// confirm (job-monitor.js restartSummary), which lists what the run reads:
// in the browser's runs the loss and distillation too, and the channel picked.
// And where the form is kept: the tab's in the browser mode, whose origin is
// the same on every visit, so an old experiment's settings do not come back
// in a new tab.
import { fileURLToPath } from "node:url";
import { afterEach, expect, it, vi } from "vitest";

const TRAINING_FORM = fileURLToPath(new URL("../../../cellmap_flow/dashboard/static/js/dashboard/finetune/training-form.js", import.meta.url));
const JOB_MONITOR = fileURLToPath(new URL("../../../cellmap_flow/dashboard/static/js/dashboard/finetune/job-monitor.js", import.meta.url));

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
  childNodes: FakeOption[] = [];  // the browser mode's notes, read before they are rewritten
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

async function setUp(mode: "server" | "browser" = "server") {
  const elements = new Map<string, FakeElement>();
  const element = (id: string): FakeElement => {
    if (!elements.has(id)) elements.set(id, new FakeElement());
    return elements.get(id)!;
  };
  element("page-data").textContent = JSON.stringify({ mode });
  element("trainChannel").options = [new FakeOption("Best match for your paint", "")];
  element("learningRate").value = "0.0001";  // the template's
  element("learningRateEase").checked = true;  // the template's
  const storage = (items: Map<string, string>) => ({ getItem: (k: string) => items.get(k) ?? null, setItem: (k: string, v: string) => items.set(k, v) });
  const stores = { local: new Map<string, string>(), session: new Map<string, string>() };
  vi.stubGlobal("window", { localStorage: storage(stores.local), sessionStorage: storage(stores.session) });
  vi.stubGlobal("document", { getElementById: element, hidden: false, addEventListener: () => {}, removeEventListener: () => {} });
  vi.stubGlobal("Option", FakeOption);
  vi.stubGlobal("fetch", async () => ({ ok: true, status: 200, json: async () => ({}) }));
  const module = (await import(/* @vite-ignore */ TRAINING_FORM)) as TrainingForm;
  return { element, module, stores, form: module.initTrainingForm() };
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
  expect(names()).toEqual(["all_mem (0)", "mito (1)", "er (2)"]);
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

it("lists in Restart's confirm what the run reads: the browser's loss and distillation, and the channel picked", async () => {
  const { element } = await setUp();
  element("trainChannel").options.push(new FakeOption("mito (1)", "1"));
  const { restartSummary } = (await import(/* @vite-ignore */ JOB_MONITOR)) as { restartSummary: (body: Record<string, unknown>, browser: boolean) => string };
  const body = {
    finetune_mode: "full", loss_type: "bce", learning_rate: 0.0001, learning_rate_schedule: "cosine", num_epochs: 20, batch_size: 1,
    tile_size: 172, distillation_lambda: 0, distillation_scope: "unlabeled", label_smoothing: 0, balance_classes: true, augment: false,
  };
  // The confirm's rows, label to value ("distillation:  0 (unlabeled)").
  const rows = (text: string) => Object.fromEntries(text.split("\n").slice(2).filter(Boolean).map((line) => line.trim().split(/:\s+/)));

  const full = rows(restartSummary(body, true));
  expect(full).toMatchObject({ training: "whole network", loss: "bce", distillation: "0 (unlabeled)", augment: "false" });
  expect(full).not.toHaveProperty("channel");  // the model trains its channels together
  expect(rows(restartSummary({ ...body, finetune_mode: "head" }, true)).distillation).toBe("0 (unlabeled)");
  expect(rows(restartSummary({ ...body, select_channel: 1 }, false)).channel).toBe("mito (1)");
});

it("sends the Learning Rate typed and its Ease down, puts both back on the next load, and refuses a rate not above 0", async () => {
  const { element, module, form } = await setUp();
  const rate = element("learningRate"), ease = element("learningRateEase");
  expect(form.trainingParams().learning_rate_schedule).toBe("cosine");
  rate.value = "3e-4";
  ease.checked = false;
  expect(form.trainingParams()).toMatchObject({ learning_rate: 0.0003, learning_rate_schedule: "constant" });
  form.save();
  rate.value = "0.0001";  // the template's, as a page loaded again has them
  ease.checked = true;
  module.initTrainingForm();
  expect([rate.value, ease.checked]).toEqual(["3e-4", false]);
  for (const bad of ["", "0", "-1e-4", "fast"]) {
    rate.value = bad;
    expect(() => form.trainingParams()).toThrow(/Learning Rate/);
  }
});

it("keeps the form in the tab's sessionStorage in the browser mode, so a new tab starts at the defaults, and in localStorage on a server", async () => {
  for (const mode of ["browser", "server"] as const) {
    vi.resetModules();  // the page's mode is read once per page load (lib/page-data.js)
    const { element, stores, form } = await setUp(mode);
    element("learningRate").value = "3e-4";
    form.save();
    const [kept, other] = mode === "browser" ? [stores.session, stores.local] : [stores.local, stores.session];
    expect(JSON.parse(kept.get("finetuneFormState")!)).toMatchObject({ learningRate: "3e-4" });
    expect(other.size).toBe(0);
  }
});
