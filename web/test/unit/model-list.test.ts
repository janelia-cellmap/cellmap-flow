// The pieces every model list of the dashboard is made of
// (cellmap_flow/dashboard/static/js/shared/model-list.js), which both modes
// run: the Models tab's Local Catalog and Hugging Face lists
// (dashboard/models/catalog.js, huggingface.js) and the Finetune tab's Which
// model to start from (dashboard/finetune/rank.js), which lists the same
// models in the same sections with the same rows. Each place's ticks are its
// own: ticking a model to score does not run it on Submit, nor the other way
// round, and each row's label is for its own box although the same model is
// listed on both tabs. In the ranking, a section's search hides the rows that
// do not match but a ticked one, which Rank scores: nothing is ranked unseen.
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const JS = "../../../cellmap_flow/dashboard/static/js/";
const module = (path: string) => fileURLToPath(new URL(JS + path, import.meta.url));

// What dom.js h() tells a child node from a string by.
class FakeNode {
  parentNode: FakeElement | null = null;
}

// A text node.
class FakeText extends FakeNode {
  constructor(public textContent: string) {
    super();
  }
}

// The few DOM calls the lists make, on plain objects: the unit tests run in
// node, which has no DOM. An attribute set on an element sets the property
// a browser derives from it (checked, value, hidden, disabled, id), and a
// selector is tags, ids, classes and :checked, joined by spaces.
class FakeElement extends FakeNode {
  childNodes: (FakeElement | FakeText)[] = [];
  attributes = new Map<string, string>();
  className = "";
  id = "";
  value = "";
  checked = false;
  disabled = false;
  hidden = false;
  open = false;
  dataset: Record<string, string> = {};
  style: Record<string, string> = {};
  listeners = new Map<string, ((event: unknown) => void)[]>();
  classList = {
    contains: (name: string) => this.className.split(/\s+/).includes(name),
    toggle: (name: string, on: boolean) => {
      const names = this.className.split(/\s+/).filter((n) => n && n !== name);
      this.className = (on ? [...names, name] : names).join(" ");
    },
  };

  constructor(public tagName: string) {
    super();
  }

  get children(): FakeElement[] {
    return this.childNodes.filter((node): node is FakeElement => node instanceof FakeElement);
  }

  get textContent(): string {
    return this.childNodes.map((node) => node.textContent).join("");
  }

  set textContent(text: string) {
    this.replaceChildren(text);
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
    if (name === "id" || name === "value") this[name] = value;
    if (name === "checked" || name === "hidden" || name === "disabled") this[name] = true;
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  append(...nodes: (FakeElement | FakeText | string)[]): void {
    for (const node of nodes) {
      const child = typeof node === "string" ? new FakeText(node) : node;
      child.parentNode?.removeChild(child);
      child.parentNode = this;
      this.childNodes.push(child);
    }
  }

  prepend(node: FakeElement): void {
    this.append(node);
    this.childNodes.unshift(this.childNodes.pop()!);
  }

  replaceChildren(...nodes: (FakeElement | FakeText | string)[]): void {
    for (const child of this.childNodes) child.parentNode = null;
    this.childNodes = [];
    this.append(...nodes);
  }

  removeChild(child: FakeElement | FakeText): void {
    this.childNodes = this.childNodes.filter((node) => node !== child);
    child.parentNode = null;
  }

  remove(): void {
    this.parentNode?.removeChild(this);
  }

  addEventListener(type: string, listener: (event: unknown) => void): void {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }

  // As a browser dispatches it, bubbling: the listeners get the element
  // it happened to as the event's target.
  dispatchEvent(event: { type: string; key?: string }): void {
    const passed = { type: event.type, key: event.key, target: this, preventDefault: () => {} };
    for (let el: FakeElement | null = this; el; el = el.parentNode) {
      for (const listener of el.listeners.get(event.type) ?? []) listener(passed);
    }
  }

  get descendants(): FakeElement[] {
    return this.children.flatMap((child) => [child, ...child.descendants]);
  }

  querySelectorAll(selector: string): FakeElement[] {
    const parts = selector.trim().split(/\s+/);
    return this.descendants.filter((el) => matches(el, parts, this));
  }

  querySelector(selector: string): FakeElement | null {
    return this.querySelectorAll(selector)[0] ?? null;
  }
}

function matchesOne(el: FakeElement, part: string): boolean {
  return (part.match(/^[a-z0-9]+|[#.:][\w-]+/gi) ?? []).every((token) =>
    token[0] === "#" ? el.id === token.slice(1)
      : token[0] === "." ? el.classList.contains(token.slice(1))
        : token === ":checked" ? el.checked
          : el.tagName === token.toLowerCase());
}

// Whether `el` matches the last of `parts` and has ancestors, under `root`,
// matching the others in order.
function matches(el: FakeElement, parts: string[], root: FakeElement): boolean {
  if (!matchesOne(el, parts[parts.length - 1])) return false;
  let rest = parts.slice(0, -1);
  for (let up = el.parentNode; up && up !== root && rest.length; up = up.parentNode) {
    if (matchesOne(up, rest[rest.length - 1])) rest = rest.slice(0, -1);
  }
  return rest.length === 0;
}

// The page: the Models tab's form, with its sections' containers, and the
// ranking's beside it, all under one root, so a lookup by id finds the first
// in the page's order as a browser's does. Any other id is a detached element
// made on first lookup.
function fakePage(data: Record<string, unknown>) {
  const root = new FakeElement("body");
  const el = (tag: string, id: string, ...children: FakeElement[]) => {
    const made = new FakeElement(tag);
    made.setAttribute("id", id);
    made.append(...children);
    return made;
  };
  root.append(
    el("form", "modelSelectionForm", el("div", "addModel", el("div", "addedModels")), el("div", "modelAccordion"), el("div", "hfModelAccordion")),
    el("div", "rankSections"),
    el("div", "rankAdd"),
  );
  const loose = new Map<string, FakeElement>();
  for (const [id, value] of Object.entries(data)) {
    loose.set(id, el("script", id));
    loose.get(id)!.textContent = JSON.stringify(value);
  }
  const document = {
    body: root,
    createElement: (tag: string) => new FakeElement(tag),
    getElementById: (id: string) => root.descendants.find((e) => e.id === id)
      ?? loose.get(id) ?? loose.set(id, el("div", id)).get(id)!,
    querySelectorAll: (selector: string) => root.querySelectorAll(selector),
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => {},
    hidden: false,
  };
  return document;
}

// The answers of the routes the lists ask: the Models tab's Hugging Face list
// and the ranking's candidates, as a server lists them.
const HF = {
  "cellmap/nuc-model": { description: "nuclei", output_voxel_size: [8, 8, 8] },
  "cellmap/mito-aff": { description: "mitochondria affinities", output_voxel_size: [16, 16, 16] },
};
const zooModel = (key: string, name: string, em: boolean) => ({
  key, id: key, name, description: `${name}, from the zoo`, tags: [], dims: "3d", em, weight_formats: ["pytorch"],
  url: `https://bioimage.io/#/?id=${key}`, trained: null,
});
const CANDIDATES = {
  success: true, hf_cached: true, zoo_cached: true,
  volume: { id: "vol-1", model_name: "mito_16nm", painted_chunks: 2 },
  candidates: [
    { name: "mito_16nm", group: "Running", ticked: true, served: true, task: "cellmap", voxel_size_nm: [16, 16, 16], dims: 3, volume_model: true },
    { name: "nuc_model", group: "Hugging Face", ticked: false, task: "huggingface", voxel_size_nm: [8, 8, 8], dims: 3, source: "cellmap/nuc-model" },
    { name: "mito_aff", group: "Hugging Face", ticked: false, task: "huggingface", voxel_size_nm: [16, 16, 16], dims: 3, source: "cellmap/mito-aff" },
    { name: "impartial_shrimp", group: "BioImage Zoo", ticked: false, em: true, dims: 3, zoo: zooModel("impartial-shrimp", "Neurons in EM", true) },
    { name: "affable_shark", group: "BioImage Zoo", ticked: false, em: false, dims: 3, zoo: zooModel("affable-shark", "Nuclei in light", false) },
  ],
};

let page: ReturnType<typeof fakePage>;

// The page in `mode`, the server's or the browser's, whose ranking lists
// `candidates`.
function setUp(mode: string, candidates: unknown) {
  vi.resetModules();
  page = fakePage({
    "page-data": { mode, default_hf_repos: [] },
    "model-catalog-data": { groups: [["cellmap", ["mito_16nm", "nuc_8nm"]]], ticked: ["mito_16nm"] },
  });
  const answers: Record<string, unknown> = { "/api/huggingface-models": HF, "/api/finetune/rank/candidates": candidates };
  vi.stubGlobal("document", page);
  vi.stubGlobal("Node", FakeNode);
  vi.stubGlobal("window", { localStorage: { getItem: () => null, setItem: () => {} } });
  vi.stubGlobal("fetch", async (url: string) => ({ ok: true, status: 200, json: async () => answers[url] ?? { success: false } }));
}

beforeEach(() => setUp("server", CANDIDATES));

afterEach(() => {
  vi.unstubAllGlobals();
});

// Until every request made so far has been answered and shown.
const settled = () => new Promise((resolve) => setTimeout(resolve, 0));

// What a row shows as its name: its label's first text.
const nameOf = (row: FakeElement) => row.querySelector("label")!.childNodes[0].textContent;

// The models tab and the ranking set up, the Hugging Face section of the
// first and the ranking's card opened, their lists in.
async function bothTabs() {
  const catalog = await import(/* @vite-ignore */ module("dashboard/models/catalog.js"));
  const huggingface = await import(/* @vite-ignore */ module("dashboard/models/huggingface.js"));
  const { initRank } = await import(/* @vite-ignore */ module("dashboard/finetune/rank.js"));
  catalog.init();
  huggingface.init();
  page.getElementById("collapse_hf").dispatchEvent({ type: "show.bs.collapse" });
  initRank();
  page.getElementById("training").className = "tab-pane active";
  const card = page.getElementById("rankSection");
  card.open = true;
  card.dispatchEvent({ type: "toggle" });
  await settled();
  const sections = page.getElementById("rankSections").children;
  const titles = sections.map((item) => item.querySelector("button")!.childNodes[0].textContent);
  // A ranking section's rows, by the name each shows.
  const rowsOf = (title: string) => {
    const section = sections[titles.indexOf(title)];
    return Object.fromEntries(section.querySelectorAll(".form-check").filter((row) => row.querySelector("label")?.getAttribute("for")?.startsWith("rankCandidate"))
      .map((row) => [nameOf(row), row]));
  };
  return { catalog, huggingface, sections, titles, rowsOf };
}

it("lists both tabs' models with the same pieces, each tab's ticks its own", async () => {
  const { catalog, huggingface, sections, titles, rowsOf } = await bothTabs();

  // The ranking's sections are the Models tab's, the running models' first;
  // the Hugging Face one has its search bar and Refresh, the zoo's its filters.
  expect(titles).toEqual(["Running", "Hugging Face", "BioImage Zoo"]);
  expect(sections[1].querySelector("#rankHfSearchBar")?.tagName).toBe("input");
  expect(["rankZooEmOnly", "rankZoo2d", "rankZoo3d"].map((id) => page.getElementById(id).tagName)).toEqual(["input", "input", "input"]);

  // Submit runs what the Models tab ticks: the running model, from its catalog.
  expect(catalog.payload()).toEqual({ selected_models: ["mito_16nm"] });
  expect(huggingface.payload()).toEqual({ selected_hf_models: [] });

  // nuc-model ticked to score: the ranking counts it, Submit does not run it.
  const nuc = rowsOf("Hugging Face")["nuc_model"].querySelector("input")!;
  nuc.checked = true;
  nuc.dispatchEvent({ type: "change" });
  expect(page.getElementById("rankTickedCount").textContent).toBe("2 ticked");
  expect(huggingface.payload()).toEqual({ selected_hf_models: [] });

  // Ticked to run on the Models tab, then unticked in the ranking: Submit
  // runs it, and Rank does not score it.
  const tabRow = page.querySelectorAll("#hfModelAccordion .hf-model-item").find((row) => nameOf(row) === "nuc-model")!;
  const tabBox = tabRow.querySelector("input")!;
  tabBox.checked = true;
  tabBox.dispatchEvent({ type: "change" });
  const ranked = rowsOf("Hugging Face")["nuc_model"].querySelector("input")!;
  ranked.checked = false;
  ranked.dispatchEvent({ type: "change" });
  expect(page.getElementById("rankTickedCount").textContent).toBe("1 ticked");
  expect(huggingface.payload()).toEqual({ selected_hf_models: ["cellmap/nuc-model"] });

  // Each label is for its own box: the page's first element of that id,
  // though the same models are listed on both tabs.
  for (const row of [tabRow, ...Object.values(rowsOf("Hugging Face")), ...Object.values(rowsOf("Running"))]) {
    expect(page.getElementById(row.querySelector("label")!.getAttribute("for")!)).toBe(row.querySelector("input"));
  }
});

it("keeps a ticked model in view when its section's search hides it, and filters the zoo as the Models tab does", async () => {
  const { rowsOf } = await bothTabs();
  const search = page.getElementById("rankHfSearchBar");
  const shown = (title: string) => Object.entries(rowsOf(title)).filter(([, row]) => !row.hidden).map(([name]) => name);
  const type = (text: string) => {
    search.value = text;
    search.dispatchEvent({ type: "input" });
  };

  expect(shown("Hugging Face")).toEqual(["nuc_model", "mito_aff"]);
  type("mito aff");
  expect(shown("Hugging Face")).toEqual(["mito_aff"]);
  // Ticked, nuc_model is scored by Rank, so it shows whatever is typed.
  type("");
  const nuc = rowsOf("Hugging Face")["nuc_model"].querySelector("input")!;
  nuc.checked = true;
  nuc.dispatchEvent({ type: "change" });
  type("mito aff");
  expect(shown("Hugging Face")).toEqual(["nuc_model", "mito_aff"]);
  // A search in one section is that section's: the running model stays.
  expect(shown("Running")).toEqual(["mito_16nm (painted for)"]);

  // The zoo's rows are the Models tab's zoo rows, its EM only ticked to start.
  expect(shown("BioImage Zoo")).toEqual(["Neurons in EM"]);
  const emOnly = page.getElementById("rankZooEmOnly");
  emOnly.checked = false;
  emOnly.dispatchEvent({ type: "change" });
  expect(shown("BioImage Zoo")).toEqual(["Neurons in EM", "Nuclei in light"]);
  expect(rowsOf("BioImage Zoo")["Neurons in EM"].querySelectorAll(".zoo-tag").map((tag) => tag.textContent)).toEqual(["3D", "pytorch"]);
});

it("gives the browser's Hugging Face list, cellmap's repo, the search bar, and lists no zoo", async () => {
  setUp("browser", {
    success: true, volume: null,
    candidates: [
      { name: "mito_aff", group: "Running", ticked: true, served: true, task: "mito", voxel_size_nm: [16, 16, 16], dims: 3 },
      { name: "nuc_model", group: "cellmap", ticked: true, task: "nuc", voxel_size_nm: [8, 8, 8], dims: 3 },
      { name: "mito_aff_ft1", group: "Finetuned", ticked: true, task: "mito", voxel_size_nm: [16, 16, 16], dims: 3, finetuned: true },
    ],
  });
  const { sections, titles } = await bothTabs();

  expect(titles).toEqual(["Running", "cellmap", "Finetuned"]);
  expect(sections[1].querySelector("#rankHfSearchBar")?.tagName).toBe("input");
  expect(page.body.querySelector("#rankZooEmOnly")).toBe(null);
});
