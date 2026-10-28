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
// On a server, a Hugging Face section shows only the models with a copy that
// runs in the browser to start, and a ticked one whatever it has.
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { FakeElement, FakeNode } from "./dashboard-dom";

const JS = "../../../cellmap_flow/dashboard/static/js/";
const module = (path: string) => fileURLToPath(new URL(JS + path, import.meta.url));

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
// `candidates` and whose Hugging Face list is `hf`, with the repos of
// `running` running (ticked on the Models tab).
function setUp(mode: string, candidates: unknown, hf: unknown = HF, running: string[] = []) {
  vi.resetModules();
  page = fakePage({
    "page-data": { mode, default_hf_repos: running },
    "model-catalog-data": { groups: [["cellmap", ["mito_16nm", "nuc_8nm"]]], ticked: ["mito_16nm"] },
  });
  const answers: Record<string, unknown> = { "/api/huggingface-models": hf, "/api/finetune/rank/candidates": candidates };
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
  initRank({ form: { gpuQueues: () => [] } });
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

  expect(shown("Hugging Face")).toEqual(["mito_aff", "nuc_model"]);
  type("mito aff");
  expect(shown("Hugging Face")).toEqual(["mito_aff"]);
  // Ticked, nuc_model is scored by Rank, so it shows whatever is typed.
  type("");
  const nuc = rowsOf("Hugging Face")["nuc_model"].querySelector("input")!;
  nuc.checked = true;
  nuc.dispatchEvent({ type: "change" });
  type("mito aff");
  expect(shown("Hugging Face")).toEqual(["mito_aff", "nuc_model"]);
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
  // Every model runs in the browser there: no box to show only those.
  expect(page.body.querySelector("#rankHfBrowser")).toBe(null);
  expect(page.body.querySelector("#hfBrowser")).toBe(null);
});

it("shows on a server the Hugging Face models with a browser copy and the ticked ones, by name or by size", async () => {
  // As models/hf_catalog.py lists them: two with a copy that runs in the
  // browser, two known to have none (one of them running), one its listing
  // says nothing of.
  const facts = (size_mb: number, browser_copy: string | null) => ({ formats: ["TorchScript"], finetune_modes: ["full"], size_mb, browser_copy });
  const hf = {
    "cellmap/setup-54": facts(56, "cellmap/onnx-web-models/setup_54"),
    "cellmap/fly-organelles": facts(3192, null),
    "cellmap/jrc-livers": facts(947, null),
    "cellmap/setup-54-2d": facts(16.5, "cellmap/onnx-web-models/setup_54_2d"),
    "cellmap/unknown": { description: "cached before its facts were listed" },
  };
  const candidates = {
    ...CANDIDATES,
    candidates: Object.entries(hf).map(([repo, meta]) => ({ name: repo.split("/")[1], group: "Hugging Face", ticked: repo === "cellmap/jrc-livers", ...meta })),
  };
  setUp("server", candidates, hf, ["cellmap/jrc-livers"]);
  const { rowsOf } = await bothTabs();
  const tabRows = () => page.querySelectorAll("#hfModelAccordion .hf-model-item");
  const onTab = () => tabRows().filter((row) => row.style.display !== "none").map(nameOf);
  const inRanking = () => Object.entries(rowsOf("Hugging Face")).filter(([, row]) => !row.hidden).map(([name]) => name);
  const change = (id: string, set: (el: FakeElement) => void) => {
    const el = page.getElementById(id);
    set(el);
    el.dispatchEvent({ type: "change" });
  };

  // Runs in the browser is ticked to start: fly-organelles is hidden, and
  // jrc-livers, running and ticked, stays, as one its listing says nothing of.
  const shownByName = ["jrc-livers", "setup-54", "setup-54-2d", "unknown"];
  expect(onTab()).toEqual(shownByName);
  expect(inRanking()).toEqual(shownByName);

  // Unticked, every one; by size, the smallest first, the one of no size last.
  change("hfBrowser", (box) => { box.checked = false; });
  change("hfSort", (select) => { select.value = "size"; });
  expect(onTab()).toEqual(["setup-54-2d", "setup-54", "jrc-livers", "fly-organelles", "unknown"]);
  change("rankHfSort", (select) => { select.value = "size"; });
  expect(inRanking()).toEqual(["setup-54-2d", "setup-54", "jrc-livers", "unknown"]);
});
