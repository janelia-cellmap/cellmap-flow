// Which model to start from: the models the page can run, ranked by how well
// each one's prediction already matches the paint of the session's volume, so
// the one finetuned is the one with the least to learn. GET
// /api/finetune/rank/candidates lists them with the volume, and POST
// /api/finetune/rank/score scores one (web/src/backend/routes/rank.ts in the
// browser, dashboard/routes/finetune/rank_routes.py on a server). The scores
// are asked for one at a time, as each can take up to a minute in the browser
// (a download, then a run over the painted chunks) and longer on a server for
// a model not running (a cluster job started for it), and each is shown as it
// comes. A server's candidate says how to build it (entry), sent back with
// each request. In both modes only the models running and the ones added by
// hand start ticked. They are listed as the Models tab lists them, made of
// the same pieces (shared/model-list.js): a section for each of the tab's
// sections a model is under (its group), with the tab's rows, the Hugging
// Face and zoo sections' search bar, Refresh and filters (a server's Hugging
// Face section showing only the models with a browser copy to start, as the
// Models tab's does, and both modes' in order of name or size), and its Add
// a model; each row tagged with what the model is (shared/model-list.js
// factTags, from the candidate's formats, finetune modes, size and browser
// copy). The ticks are the ranking's own, not the Models tab's: a ticked
// model stays in view whatever its section's search and filters, and Rank
// scores every ticked one. Each row of the ranking gives what its score was
// read at, as the controls it sets take it (portable/rank.py score_row): the
// threshold, a probability, and the input normalization it was read through
// (input_norm: the model's own, or the session's when that matched better,
// maybe stretched to the raw under the paint). Show serves a model and adds
// its layer, as the Models tab's Add does; Use this model (POST
// /api/finetune/rank/use) makes one the model to finetune: that input
// normalization the session's, the model served, a new volume made for it
// with the paint carried onto its grid, that volume the session's and its
// paint layer in the viewer. The model picker (models.js pickModel) and the
// Annotation Crops panel's volume buttons (crops.js sessionVolumeChanged)
// follow, and a binary model's row's channel (channel_index, the channel
// that matched best) becomes its Channel in the training form
// (training-form.js pickChannel), shown once the model is picked if it
// trains one channel at a time. Rank itself fills the Channel of the model
// the form shows, when none is picked (fillChannel): that is the model
// painted for, which has no Use this model.
//
// Fit to Paint, beside Seed from Prediction, does the same for the model
// that seeds the paint (Label it's Prediction from) without a new volume:
// it scores that model with the same route, and sets its threshold and,
// when it is not the session's as it stands, the input normalization its
// score was read through (POST /api/finetune/rank/input-norm); its
// Channel only when the form shows one for that model with nothing picked
// (fillChannel), so a channel the user picked stays. Both it and Use this
// model mark Seed from Prediction's threshold as set by ranking, and say on
// the line under Label it's buttons what they set.
import { h, setBusy } from "../../lib/dom.js";
import { announce, CHAIN_CHANGED_EVENT, listen, MODEL_LIST_CHANGED_EVENT, MODELS_CHANGED_EVENT } from "../../lib/events.js";
import { isBrowserMode } from "../../lib/page-data.js";
import { searchText } from "../../lib/search.js";
import { readJSON, writeJSON } from "../../lib/stored.js";
import { addModelField, listControls, modelRow, modelSection, neededField, notice, zooRow } from "../../shared/model-list.js";
import { sessionVolumeChanged } from "./crops.js";
import { pickModel } from "./models.js";
import { getAnswer, offerNewVolume, postAnswer } from "./requests.js";
import { channelLabel, fillChannel, pickChannel } from "./training-form.js";

// A score this close to the best one is marked about as good as it: two
// points of matching are not worth choosing a model by.
export const ABOUT_AS_GOOD = 0.02;
// With fewer painted voxels than this behind the best score, scores a few
// points apart are as good as tied, and the table says so.
export const LITTLE_PAINT = 1000;

// The table's rows from the score answers so far (each with the model's
// name, and success false with an error for one that could not be scored):
// the models scored, the best match of the paint first (equal scores keep
// the order they came in), then the others in the order they came. The
// first is marked best, and each within ABOUT_AS_GOOD of it tied (and a
// hair over: 0.81 - 0.79 is not exactly 0.02 in floating point).
// littlePaint: whether the best was scored on fewer than LITTLE_PAINT voxels.
export function rankResults(answers) {
  const scored = answers.filter(isScored).sort((a, b) => b.matches - a.matches);
  const failed = answers.filter((answer) => !isScored(answer));
  const best = scored[0];
  const rows = scored
    .map((answer, i) => ({ ...answer, best: i === 0, tied: i > 0 && best.matches - answer.matches <= ABOUT_AS_GOOD + 1e-9 }))
    .concat(failed.map((answer) => ({ ...answer, best: false, tied: false })));
  return { rows, littlePaint: Boolean(best) && best.scored_voxels < LITTLE_PAINT };
}

// The text a section's search reads in `candidate`'s row (lib/search.js,
// the rule every model list's search follows): the model's name, task,
// channels and voxel size ("16nm", which "16 nm" finds too), its "2d" or
// "3d", and "finetuned" for a finetune. A zoo model's row is searched as
// the Models tab's zoo list searches it (shared/model-list.js zooRow).
export function candidateText(candidate) {
  return searchText(
    candidate.name, candidate.task, ...(candidate.channels || []),
    ...(candidate.voxel_size_nm || []).map((size) => `${size}nm`),
    candidate.dims && `${candidate.dims}d`, candidate.finetuned && "finetuned",
  );
}

function isScored(answer) {
  return Boolean(answer.success) && Number.isFinite(answer.matches);
}

// Where a score's input normalization came from (portable/rank.py
// norm_tries's `from`), in the row's words.
const NORM_FROM = {
  declared: "its own",
  convention: "CellMap convention",
  assumed: "assumed [−1, 1]",
  session: "yours",
  none: "none",
};

// A scored row's settings, as columns (each a fixed width, so they line
// up from row to row): the threshold Seed from Prediction takes, 0.5 when
// the model's own boundary fits the paint best, else with the match at 0.5;
// the input normalization it was read through, where it came from and its
// steps, stretched to the raw under the paint when that scored clearly
// better, then with the match unstretched, and what the others tried
// matched (cellmap_flow/portable/rank.py score_row); and the postprocessing
// its layer needs, when it needs any: the sigmoid added or taken away and
// the threshold the paint is fitted best at (needed_postprocess).
function settingsCells(row) {
  const own = row.cutoff === 0;
  const stretched = Array.isArray(row.window);
  return h("div", { class: "d-flex flex-wrap column-gap-3 small" },
    h("span", { style: { minWidth: "11rem" }, title: "The probability above which the prediction fits your paint best, as Seed from Prediction's threshold takes it; 0.5 is the model's own boundary" },
      "Threshold ", h("strong", {}, String(row.threshold)),
      own ? h("span", { class: "text-muted" }, " (model's own)") : h("span", { class: "text-muted" }, ` (${percent(row.matches_at_zero)} at 0.5)`)),
    h("span", { title: "The input normalization the model was read through: its own (what it declares, or the CellMap convention for a CellMap model), or yours from the Processing tab when that matched better; Use this model sets it" },
      "Input ", h("strong", {}, NORM_FROM[row.input_norm_from] || "as normalized"),
      Array.isArray(row.input_norm) && row.input_norm.length ? h("span", { class: "text-muted" }, ` ${stepsText(row.input_norm)}`) : "",
      stretched ? h("span", { class: "text-muted" }, `, stretched (${percent(row.matches_without_window)} unstretched)`) : "",
      (row.input_norm_others || []).length ? h("span", { class: "text-muted" }, `; ${othersText(row.input_norm_others)}`) : ""),
    Array.isArray(row.postprocess) && h("span", { title: `The postprocessing the model's output needs to show right, which Use this model sets: a sigmoid added for an output with no activation (logits) or taken away from one that has one already, and the threshold your paint is fitted best at. ${activationText(row)}` },
      "Output ", h("strong", {}, outputWords(row.postprocess))));
}

// Where the output activation a row was read by came from (the score's
// output_class_from and output_op, its server's model_info's): the
// network's last op, which says it for certain, or a guess from the model's
// output on noise (the browser: its manifest's words).
function activationText(row) {
  const op = row.output_op;
  if (row.output_class_from === "graph" && op) return `Its network ends in ${/^[AEIOU]/.test(op) ? "an" : "a"} ${op}.`;
  if (row.output_class_from === "probe") return "Its network does not say what its output is: read off its output on noise, a guess.";
  if (row.output_class_from === "manifest") return "Its network does not say what its output is: read from its manifest's words, a guess.";
  return "What its output is, neither its network nor its output on noise says.";
}

// What a score that is right on one class only does, in words: a model
// that calls (almost) everything object finds every painted object and
// matches no better than guessing (objects 99%, background 34%: 58%), and
// "objects 99%" alone read as good. Null when both are right at least half
// the time.
function lopsided(row) {
  if (Number.isFinite(row.background_right) && row.background_right < 0.5) return "calls most of your background object too";
  if (Number.isFinite(row.objects_found) && row.objects_found < 0.5) return "misses most of your objects";
  return null;
}

// What the other input normalizations tried matched: "none 51%".
function othersText(others) {
  return others.map((o) => `${NORM_FROM[o.from] || o.from} ${percent(o.matches)}`).join(", ");
}

// A row's postprocessing (needed_postprocess's), in short: "+ sigmoid, cut
// at 0.731", "cut at 0.731" for an output that has its activation, or "no
// second sigmoid" when the steps are the chain's with its sigmoid taken away.
function outputWords(steps) {
  const cut = steps.find((step) => step.name === "ThresholdPostprocessor");
  const words = [steps.some((step) => step.name === "SigmoidPostprocessor") && "+ sigmoid", cut && `cut at ${cut.threshold}`].filter(Boolean);
  return words.length ? words.join(", ") : "no second sigmoid";
}

// The rest of a binary model's channels beside its best (channel_matches,
// the best first): the two closest, by name, number and match ("next: mito
// (2) 85%, er (3) 60%"), or that none does better than guessing.
function runnersUp(row) {
  const others = (row.channel_matches || []).slice(1);
  if (!others.length) return "";
  if (others[0].matches <= 0.505) return "every other channel 50% or less";
  return `next: ${others.slice(0, 2).map((c) => `${channelLabel(c.channel, c.channel_index, false)} ${percent(c.matches)}`).join(", ")}`;
}

// Normalizer steps in a few words each: a MinMaxNormalizer as the raw's
// range it reads as 0 to 1, a LambdaNormalizer as its expression, any other
// by its name.
function stepsText(steps) {
  return steps.map((step) => {
    if (step.name === "MinMaxNormalizer") return `${levels([Number(step.min_value ?? 0), Number(step.max_value ?? 255)])} to 0–1`;
    if (step.name === "LambdaNormalizer") return step.expression;
    return step.name.replace(/Normalizer$/, "");
  }).join(" → ");
}

// A raw window [lo, hi] as grey levels: whole ones for a raw of hundreds of
// levels, three significant digits for a raw of a few (a float raw's 0–1).
function levels(window) {
  return window.map((v) => String(Math.abs(v) >= 100 ? Math.round(v) : Number(v.toPrecision(3)))).join("–");
}

function percent(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "–";
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

// A candidate's line beside its name: what it predicts, at what voxel size,
// in 2D or 3D, and whether it is a finetune.
function describe(candidate) {
  const size = candidate.voxel_size_nm || [];
  const nm = size.length && size.every((s) => s === size[0]) ? `${size[0]} nm` : `${size.join("×")} nm`;
  return [candidate.task, size.length && nm, candidate.dims && `${candidate.dims}D`, candidate.finetuned && "finetuned"]
    .filter(Boolean).join(", ");
}

// The ranking's sections, one for each group the candidates are under (the
// routes' RUNNING, HUGGING_FACE, ZOO and ADDED, a server's Local Catalog
// groups, and the browser's GROUPS), in the Models tab's order, which the
// routes list the candidates in; null is where the groups named nowhere
// else go, in the order listed: a server's Local Catalog's. `hf` is the
// group of the Models tab's Hugging Face list, whose section has its search
// bar and Refresh: the browser's is cellmap's own repo, where a server's
// "cellmap" is a Local Catalog group. That section and the zoo's show even
// while they list nothing, for their Refresh.
const ZOO_GROUP = "BioImage Zoo";
const SERVER_GROUPS = { order: ["Running", null, "Hugging Face", ZOO_GROUP, "Added"], hf: "Hugging Face" };
const BROWSER_GROUPS = { order: ["Running", "cellmap", "Added", "Finetuned", null], hf: "cellmap" };
// The Models tab's Hugging Face and zoo lists (GET, and POST .../refresh to
// fetch them again), which a listing of the candidates reads only as those
// cached them: a server's ranking never waits on the Hub or bioimage.io.
const LISTS = { hf: "/api/huggingface-models", zoo: "/api/bioimage-models" };
// What their sections say until then, as the Models tab's do.
const NOT_LOADED = { hf: "Click to load Hugging Face models...", zoo: "Click to load BioImage Model Zoo models..." };
// What rank/add takes as voxel_size of what a model needs (the route's
// VOXEL_NEEDS); anything else only the Models tab's Add a model asks for.
const VOXEL_NEEDS = ["voxel_size", "input_voxel_size"];

// Whether the card is open, kept in this browser as the patch tools'
// sections are (index.js rememberOpenSections); closed until opened.
const OPEN_KEY = "finetuneRankOpen";
// The threshold Use this model or Fit to Paint last set, and for which
// model: {threshold, model}, kept in this browser for the field's mark.
const MARK_KEY = "cellmap_flow.seed_threshold_by_ranking";

// form: the training form (training-form.js), for its GPU queues.
export function initRank({ form }) {
  const section = document.getElementById("rankSection");
  const tickAll = document.getElementById("rankTickAll");
  const tickNone = document.getElementById("rankTickNone");
  // The sections, and Add a model under them.
  const sectionList = document.getElementById("rankSections");
  const groups = isBrowserMode() ? BROWSER_GROUPS : SERVER_GROUPS;
  // Ticked only shows the models Rank scores, and the count beside it says
  // how many. Tick none unticks every one, in every section.
  const tickedOnly = document.getElementById("rankTickedOnly");
  const tickedCount = document.getElementById("rankTickedCount");
  const rankBtn = document.getElementById("rankBtn");
  const stopBtn = document.getElementById("rankStopBtn");
  const status = document.getElementById("rankStatus");
  const results = document.getElementById("rankResults");
  const rows = document.getElementById("rankRows");
  const littlePaint = document.getElementById("rankLittlePaint");
  // Label it's: the model seeding the paint, the threshold and settings a
  // score sets (label-it.js reads and keeps them), the threshold's mark,
  // Fit to Paint and the line saying what was set.
  const seedModel = document.getElementById("seedModel");
  const seedThreshold = document.getElementById("seedThreshold");
  const seedSettings = document.getElementById("seedSettings");
  const thresholdMark = document.getElementById("seedThresholdByRanking");
  const fitBtn = document.getElementById("fitToPaintBtn");
  const fitStatus = document.getElementById("fitToPaintStatus");

  let candidates = [];
  // The models unticked, by hand or as listed (ticked false): one listed
  // later starts as its listing says.
  const unticked = new Set();
  const seen = new Set();
  // A section for each group, made once and kept, so it stays open or closed
  // as it was left and its search keeps what was typed while the list is
  // shown again: {section, controls, list, kind}, kind "hf" or "zoo" for the
  // Models tab's lists of those, with their controls. Each model's row as
  // shown last, by name. Whether the listing had the Models tab's Hugging
  // Face and zoo lists cached (a browser's always has its catalog), and why
  // one could not be fetched.
  const sections = new Map();
  let rowsByName = new Map();
  let cached = { hf: true, zoo: true };
  const listErrors = {};
  // The ranking shown: its score answers in the order they came, the models
  // it scores by name (for Show), the volume it was painted on, and its
  // status line ({ text, tone }), which stays while its table does.
  let answers = [];
  let ranked = new Map();
  let rankedOn = null;
  let summary = null;
  let running = false;
  let stopping = false;
  let listed = false;
  // The model Use this model is making the one to finetune, while it does;
  // and the one it made, with its volume, while that is the session's.
  let switching = null;
  let training = null;
  // Whether Fit to Paint is scoring and setting, which Rank and Use wait for
  // (and it for them): one score at a time. And whether the last listing
  // found paint to rank on, for Rank once Fit to Paint is done.
  let fitting = false;
  let ready = false;

  function say(text, tone = "muted") {
    status.textContent = text;
    status.className = `d-block text-${tone}`;
  }

  // The line under Label it's buttons, for what a score set or why it could not.
  function sayFit(text, tone = "muted") {
    fitStatus.hidden = false;
    fitStatus.textContent = text;
    fitStatus.className = `d-block mt-1 text-${tone}`;
  }

  // What Use this model or Fit to Paint set from `row`, the model's score:
  // the threshold, the channel when one was set (`channel`, its name), and
  // the input normalization the score was read through (inputText).
  function fittedText(name, row, channel) {
    return `${name} fitted to your paint (${percent(row.matches)} match): Seed from Prediction's threshold is ${row.threshold}${channel ? `, ${channelSet(channel)}` : ""}${changesOutput(row) ? `, ${outputText(row)}` : ""}, and ${inputText(row)}.`;
  }

  // Whether `row` carries postprocessing to set: the sigmoid its layer needs
  // added or taken away, and the threshold (portable/rank.py
  // needed_postprocess).
  function changesOutput(row) {
    return Array.isArray(row.postprocess);
  }

  // The postprocessing a score set, in words: the sigmoid its output needs
  // (or a second one taken away), and the threshold the paint is fitted
  // best at, in its output's units after that (portable/rank.py
  // needed_postprocess: on a tanh's [-1, 1], 2p - 1 for the row's p).
  function outputText(row) {
    const cut = row.postprocess.find((step) => step.name === "ThresholdPostprocessor");
    const sigmoid = row.postprocess.some((step) => step.name === "SigmoidPostprocessor");
    if (cut) return `its layer shows ${sigmoid ? "the sigmoid of its output" : "its output"} cut at ${cut.threshold}`;
    return sigmoid ? "a sigmoid turns its output into probabilities" : "the second sigmoid on its output is taken away";
  }

  // The chain parts a score sets, as Use this model and Fit to Paint send them.
  function chainParts(row) {
    return { ...(changesInput(row) ? { input_norm: row.input_norm } : {}), ...(changesOutput(row) ? { postprocess: row.postprocess } : {}) };
  }

  // Whether setting `row`'s input normalization changes the session's: it
  // does unless the score was read through the session's as it stands (or
  // through none, with none set).
  function changesInput(row) {
    return Array.isArray(row.input_norm) && (Array.isArray(row.window) || !["session", "none"].includes(row.input_norm_from));
  }

  // The input normalization a score set, in words, for every model, or that
  // it was left as it was.
  function inputText(row) {
    if (!changesInput(row)) return "the input normalization is left as it was";
    const from = { declared: "its declared one", convention: "the CellMap convention", assumed: "the assumed [−1, 1]", session: "yours" }[row.input_norm_from] || "the one it was read through";
    return `the input normalization is now ${from}${Array.isArray(row.window) ? `, stretched to the raw under your paint` : ""} (${stepsText(row.input_norm)}), for every model`;
  }

  // Why a score's model fits the paint, in words: how well it matches, on
  // which channel ahead of which, through which normalization ahead of
  // which.
  function whyText(row) {
    const channel = row.channel ? ` on its ${channelLabel(row.channel, row.channel_index)}${runnersUp(row) ? ` (${runnersUp(row)})` : ""}` : "";
    const others = (row.input_norm_others || []).length ? ` (${othersText(row.input_norm_others)})` : "";
    return `${percent(row.matches)} match${channel}, read through ${NORM_FROM[row.input_norm_from] || "its input normalization"}${others}`;
  }

  // A channel set from a score, in words: a binary model's score is read on
  // its best channel (channel_index), which Use this model makes the model's
  // Channel (pickChannel) and Fit to Paint fills in when none is picked
  // (fillChannel): the channel it trains and Seed from Prediction reads. A
  // model of one channel trains that one anyway; an affinity model's
  // channels are scored together, and set none.
  function channelSet(channel) {
    return `the channel it trains and seeds from is ${channel}`;
  }

  // The threshold a score fitted, made Seed from Prediction's (label-it.js
  // keeps it with the seed settings on its change) and marked as set by
  // ranking, with the settings opened to show it. The mark is kept in this
  // browser with the value it marks, and shown while the field holds that
  // value: typing another takes it away.
  function installThreshold(name, threshold) {
    if (!Number.isFinite(threshold)) return;
    writeJSON(MARK_KEY, { threshold, model: name });
    seedThreshold.value = threshold;
    seedThreshold.dispatchEvent(new Event("change"));
    seedSettings.open = true;
  }

  function showMark() {
    const mark = readJSON(MARK_KEY, null);
    const marks = Boolean(mark) && seedThreshold.value !== "" && Number(seedThreshold.value) === Number(mark.threshold);
    thresholdMark.hidden = !marks;
    if (marks) thresholdMark.title = `Set by ranking for ${mark.model}: the threshold that fits your paint best. Changing it takes the mark away.`;
  }

  // The candidates and the volume as the page has them now. Resolves to
  // { ready, message, tone }: whether there is paint to rank on, and what
  // the status line says about it. A ranking of another volume's paint is
  // cleared, and the rows shown take the models' served state from the list.
  async function listCandidates() {
    let d;
    try {
      d = await getAnswer("/api/finetune/rank/candidates");
    } catch (e) {
      return { ready: false, message: `Could not list the models: ${e}`, tone: "danger" };
    }
    if (!d.success) return { ready: false, message: `Could not list the models: ${d.error}`, tone: "danger" };
    listed = true;
    candidates = d.candidates || [];
    cached = { hf: d.hf_cached !== false, zoo: d.zoo_cached !== false };
    for (const candidate of candidates) {
      if (!seen.has(candidate.name) && candidate.ticked === false) unticked.add(candidate.name);
      seen.add(candidate.name);
    }
    const volume = d.volume;
    // (Not while Use this model makes the new volume, which keeps the ranking.)
    if (!running && switching === null && answers.length && rankedOn !== (volume && volume.id)) {
      answers = [];
      summary = null;
    }
    if (training && training.volumeId !== (volume && volume.id)) training = null;
    for (const candidate of candidates) if (ranked.has(candidate.name)) ranked.set(candidate.name, candidate);
    showCandidates();
    showResults();
    if (!volume) {
      return { ready: false, message: "No annotation volume yet: make one with New Volume (Annotation Crops tab), paint on it, then rank.", tone: "muted" };
    }
    if (!volume.painted_chunks) {
      return { ready: false, message: `Nothing is painted in ${volume.id} yet: paint background (1) and objects (2 and up), then rank.`, tone: "muted" };
    }
    if (!candidates.length) return { ready: false, message: "No models to rank.", tone: "muted" };
    const paintedFor = volume.model_name ? `painted for ${volume.model_name}` : "painted before a model was chosen";
    return { ready: true, volume, message: `${plural(volume.painted_chunks, "painted chunk")} in ${volume.id}, ${paintedFor}.`, tone: "muted" };
  }

  // The status line and Rank from a listing; a ranking running, or one
  // shown, keeps its own line. Rank waits while Use this model or Fit to
  // Paint runs.
  function showReadiness(state) {
    if (running) return;
    ready = state.ready;
    rankBtn.disabled = !state.ready || switching !== null || fitting;
    if (state.ready && answers.length && summary) say(summary.text, summary.tone);
    else say(state.message, state.tone);
  }

  // A listing for the status line, one at a time.
  let refreshing = null;
  function refresh() {
    refreshing = refreshing || listCandidates().then(showReadiness).finally(() => { refreshing = null; });
    return refreshing;
  }

  // A listing begun after what was just changed (a model added, a list
  // fetched): one under way may have begun before it.
  async function relist() {
    if (refreshing) await refreshing;
    return refresh();
  }

  function isTicked(candidate) {
    return !unticked.has(candidate.name);
  }

  // Whether `candidate`'s row (`row`, its data the section's controls
  // match) shows: it is ticked (a ticked one stays in view, as the Models
  // tab's zoo list keeps it, so nothing is ranked unseen), or, but under
  // Ticked only, it passes its section's search and filters when the
  // section has them (the Hugging Face and zoo sections).
  function visible(candidate, row) {
    if (isTicked(candidate)) return true;
    if (tickedOnly.checked) return false;
    const { controls } = sections.get(candidate.group) || {};
    return !controls || controls.matches(row.dataset);
  }

  // The models shown: their rows in view, in a section open.
  function shown() {
    return candidates.filter((candidate) => {
      const row = rowsByName.get(candidate.name);
      const { section: shownIn } = sections.get(candidate.group) || {};
      return row && !row.hidden && shownIn && !shownIn.item.hidden && shownIn.isOpen();
    });
  }

  // The groups to show a section for, in the Models tab's order: each one a
  // candidate is under, and the Hugging Face and zoo lists' even when they
  // list nothing yet.
  function groupsShown() {
    const listed = [...new Set(candidates.map((candidate) => candidate.group))];
    const others = listed.filter((group) => !groups.order.includes(group));
    return groups.order.flatMap((group) => (group === null ? others : [group]))
      .filter((group) => listed.includes(group) || group === groups.hf || group === ZOO_GROUP);
  }

  // `group`'s section, made the first time: open when a model in it is
  // ticked, so the ones Rank scores are in view; for the Models tab's
  // Hugging Face and zoo lists, with their search bar and Refresh, and the
  // zoo's filters, and opening it fetches the list when the listing had it
  // not cached, as opening it on the Models tab does.
  function sectionFor(group) {
    if (sections.has(group)) return sections.get(group);
    const kind = group === groups.hf ? "hf" : group === ZOO_GROUP ? "zoo" : null;
    const made = modelSection({
      id: `rank_${sections.size}`, title: group,
      open: candidates.some((candidate) => candidate.group === group && isTicked(candidate)),
      onOpen: kind && (() => {
        if (!cached[kind]) fetchList(kind, false);
      }),
    });
    const controls = kind === "hf"
      ? listControls({
        id: "rankHf", placeholder: "Search: mito aff, 16 nm, 2d", title: "Every word typed, in the model's name, task, channels, voxel size or its 2D/3D",
        refreshTitle: "Refresh from Hugging Face", onRefresh: () => fetchList("hf", true), onChange: showCandidates,
        browser: !isBrowserMode(), sort: true,
      })
      : kind === "zoo"
        ? listControls({
          id: "rankZoo", zoo: true, placeholder: "Search models...", title: "Every word typed, in the name, description, tags or 2D/3D",
          refreshTitle: "Refresh from bioimage.io", onRefresh: () => fetchList("zoo", true), onChange: showCandidates,
        })
        : null;
    const list = h("div", { class: kind === "zoo" ? "zoo-model-list" : null });
    made.body.append(...(controls ? [controls.element] : []), list);
    sections.set(group, { section: made, controls, list, kind });
    return sections.get(group);
  }

  // A model's row, as the Models tab's: a zoo model's as its zoo list has it,
  // with why it cannot be ranked under it when nothing says its voxel size;
  // any other's its name, with what it predicts at what voxel size beside
  // and what it is as tags (the candidate's facts). The volume's model says
  // it is the one painted for.
  function candidateRow(candidate, id) {
    const paintedFor = candidate.volume_model ? " (painted for)" : "";
    const options = {
      id, checked: isTicked(candidate), disabled: running,
      onTick: (on) => {
        if (on) unticked.delete(candidate.name);
        else unticked.add(candidate.name);
        showCandidates();
      },
    };
    if (candidate.zoo) {
      return zooRow(candidate.zoo, {
        ...options, name: candidate.zoo.name + paintedFor,
        below: candidate.error ? h("div", { class: "zoo-trained" }, candidate.error) : null,
      });
    }
    return modelRow({
      ...options, value: candidate.name, name: candidate.name + paintedFor, note: describe(candidate),
      title: candidate.source || `${candidate.repo}/${candidate.folder}`, search: candidateText(candidate), facts: candidate,
    });
  }

  // The sections, each with its models' rows, in the order its controls
  // choose: the rows not shown hidden, the count of those ticked in its
  // header, the count of those shown in its controls (and of those hidden
  // for having no browser copy), and under Ticked only the sections with
  // none ticked hidden. A section is put back in its place only when the
  // order changed: moved, its search bar would lose what is being typed.
  function showCandidates() {
    tickAll.disabled = tickNone.disabled = running;
    tickedCount.textContent = `${candidates.filter(isTicked).length} ticked`;
    const shownGroups = groupsShown().map(sectionFor);
    const items = shownGroups.map(({ section: made }) => made.item);
    if (items.length !== sectionList.children.length || items.some((item, i) => sectionList.children[i] !== item)) {
      sectionList.replaceChildren(...items);
    }
    rowsByName = new Map();
    candidates.forEach((candidate, i) => rowsByName.set(candidate.name, candidateRow(candidate, `rankCandidate${i}`)));
    for (const [group, { section: made, controls, list, kind }] of sections) {
      const members = candidates.filter((candidate) => candidate.group === group);
      const listed = members.map((candidate) => {
        const row = rowsByName.get(candidate.name);
        row.hidden = !visible(candidate, row);
        return row;
      });
      const rows = controls ? controls.sorted(listed) : listed;
      const ticked = members.filter(isTicked).length;
      const inView = rows.filter((row) => !row.hidden).length;
      made.count(ticked ? `${ticked} ticked` : "");
      if (controls) controls.count(inView, rows.length, rows.filter((row) => row.hidden && !tickedOnly.checked && controls.noCopy(row.dataset)).length);
      made.item.hidden = tickedOnly.checked && !ticked;
      const empty = !rows.length && (listErrors[kind] ? notice(listErrors[kind], true)
        : notice(kind && !cached[kind] ? NOT_LOADED[kind] : "No models found."));
      const none = rows.length && !inView && notice("No model matches the search.");
      list.replaceChildren(...rows, ...[empty, none].filter(Boolean));
    }
  }

  // Add a model, as the Models tab's does, to this list (POST
  // /api/finetune/rank/add resolves the reference and keeps the model in the
  // session's list), ticked, without starting it. One whose voxel size
  // nothing says is asked for it, as the Models tab's asks, and added with
  // it (`voxelSize`); one that needs more is the Models tab's to add.
  const addField = addModelField({
    id: "rankAddRef", button: "Add",
    placeholder: isBrowserMode() ? "Hugging Face repo: owner/repo, owner/repo/folder" : "Hugging Face repo, zoo model, path...",
    title: "Any model the Models tab's Add a model takes but a Cellpose model; it joins the list, ticked, without starting",
    onSubmit: (ref) => addModel(ref),
  });
  document.getElementById("rankAdd").append(addField.element);

  async function addModel(ref, voxelSize) {
    if (running) return;
    setBusy(addField.button, true, "Adding...");
    try {
      const d = await postAnswer("/api/finetune/rank/add", { ref, ...(voxelSize ? { voxel_size: voxelSize } : {}) });
      if (!d.success) {
        const needs = d.needs || [];
        if (needs.length && needs.every((key) => VOXEL_NEEDS.includes(key))) {
          addField.show({
            summary: d.error, fields: [neededField(needs[0])], action: "Add",
            actionTitle: "Add it with this voxel size, ticked, without starting it",
            onAction: (values) => addModel(ref, values[needs[0]].trim()),
          });
          return;
        }
        addField.hide();
        say(`Could not add ${ref}: ${d.error || "not a model reference"}`, "danger");
        return;
      }
      const name = (d.candidate || d).name;
      unticked.delete(name);
      addField.hide();
      addField.input.value = "";
      await relist();
      say(`Added ${name}, ticked: Rank scores it with the others.`, "success");
    } catch (e) {
      say(`Could not add ${ref}: ${e.message || e}`, "danger");
    } finally {
      setBusy(addField.button, false);
    }
  }

  // The Models tab's Hugging Face or zoo list (`kind`) fetched as that tab's
  // section fetches it, from its cache or, `fresh` (its Refresh), from the
  // Hub or bioimage.io again (the same routes in both modes), then this
  // list, which lists what it fetched.
  async function fetchList(kind, fresh) {
    if (running) return;
    const { section: made } = [...sections.values()].find((s) => s.kind === kind);
    made.loading(true);
    try {
      const d = await (fresh ? postAnswer(`${LISTS[kind]}/refresh`) : getAnswer(LISTS[kind]));
      listErrors[kind] = d && d.error ? `Could not list them: ${d.error}` : null;
    } catch (e) {
      listErrors[kind] = `Could not list them: ${e.message || e}`;
    }
    try {
      await relist();
    } finally {
      made.loading(false);
    }
  }

  function showResults() {
    // Fit to Paint waits while a ranking or Use this model runs.
    fitBtn.disabled = running || switching !== null || fitting;
    const { rows: table, littlePaint: little } = rankResults(answers);
    results.hidden = !table.length;
    littlePaint.hidden = !little;
    rows.replaceChildren(...table.map(resultRow));
  }

  // A model's row: its name and badges beside Use this model (a scored model
  // the volume is not for) and Show, its scores on a line under them, its
  // threshold and input window under those, then what they were scored on.
  function resultRow(row) {
    const candidate = ranked.get(row.name);
    const scored = isScored(row);
    const details = [
      row.channel && `${channelLabel(row.channel, row.channel_index)}${runnersUp(row) ? ` · ${runnersUp(row)}` : ""}`,
      scored && `${Number(row.scored_voxels || 0).toLocaleString()} voxels (${row.object_voxels} object, ${row.background_voxels} background)`,
      row.note,
      Number.isFinite(row.seconds) && `${row.seconds.toFixed(1)} s`,
    ].filter(Boolean).join(", ");
    return h("li", {
      class: "py-1 px-2 mb-1 rounded",
      style: row.best ? { background: "var(--accent-muted)" } : { borderBottom: "1px solid var(--border)" },
    },
      h("div", { class: "d-flex flex-wrap align-items-start gap-2" },
        h("div", { class: "flex-grow-1 text-break" },
          row.name, candidate && candidate.volume_model ? " (painted for)" : "",
          row.best && h("span", { class: "badge bg-success ms-1" }, "best"),
          row.tied && h("span", { class: "badge bg-secondary ms-1" }, "about as good"),
          training && training.name === row.name && h("span", { class: "badge bg-info ms-1" }, "training this")),
        // The buttons go under the name when the column is too narrow for both.
        h("div", { class: "d-flex gap-1 flex-shrink-0 ms-auto" },
          candidate && scored && (candidate.volume_model ? settingsButton(candidate, row) : useButton(candidate, row)),
          candidate && showButton(candidate))),
      scored
        ? h("div", {}, h("strong", {}, `${percent(row.matches)} match`),
          ` · objects ${percent(row.objects_found)} · background ${percent(row.background_right)}`,
          lopsided(row) && h("span", { class: "text-warning" }, ` · ${lopsided(row)}`))
        : h("div", { class: "small text-danger" }, row.error || "No score in the answer"),
      scored && settingsCells(row),
      details && h("small", { class: "d-block text-muted" }, details));
  }

  // None for a server's model that is not running and has no entry to start
  // it from: one of the session's, ticked on the Models tab to run it.
  function showButton(candidate) {
    if (candidate.cluster && !candidate.served && !candidate.entry) return "";
    const button = h("button", {
      type: "button", class: "btn btn-outline-secondary btn-sm", disabled: candidate.served,
      title: candidate.served ? "Served already: its layer is in the viewer" : "Serve this model and add its layer to the viewer",
    }, candidate.served ? "Shown" : "Show");
    button.addEventListener("click", () => showModel(candidate, button));
    return button;
  }

  function showModel(candidate, button) {
    setBusy(button, true);
    postAnswer("/api/models/add", { entry: candidate.entry || { name: candidate.name, repo: candidate.repo, folder: candidate.folder } })
      .then((d) => {
        if (!d.success) throw new Error(d.error);
        candidate.served = true;
        showResults();
        announce(MODELS_CHANGED_EVENT);
      })
      .catch((e) => {
        setBusy(button, false);
        say(`Could not show ${candidate.name}: ${e.message || e}`, "danger");
      });
  }

  // The best row's is the card's suggestion, filled.
  function useButton(candidate, row) {
    const button = h("button", {
      type: "button", class: `btn btn-sm ${row.best ? "btn-primary" : "btn-outline-secondary"}`, disabled: running || switching !== null || fitting,
      title: "Finetune this model instead: your paint is copied to a new volume made for it, which becomes the session's, and the current volume is kept",
    }, "Use this model");
    if (switching === candidate.name) setBusy(button, true, "Copying paint...");
    button.addEventListener("click", () => useModel(candidate, row));
    return button;
  }

  // The row of the model the volume is painted for has no Use this model
  // (there is no other volume to make): this sets what its score was read
  // at instead, as Use this model would (applyRow).
  function settingsButton(candidate, row) {
    const button = h("button", {
      type: "button", class: `btn btn-sm ${row.best ? "btn-primary" : "btn-outline-secondary"}`, disabled: running || switching !== null || fitting,
      title: "Set what this model was scored at: its input normalization, the sigmoid its output needs, Seed from Prediction's threshold and its Channel. Your volume is painted for it already",
    }, "Use these settings");
    button.addEventListener("click", async () => {
      if (running || switching !== null || fitting) return;
      fitting = true;
      showResults();
      try {
        await applyRow(candidate.name, row, true);
      } catch (e) {
        sayFit(`Could not set ${candidate.name}'s settings: ${e.message || e}`, "danger");
      } finally {
        fitting = false;
        showResults();
      }
    });
    return button;
  }

  // Sets what `row` (model `name`'s score) was read at, with no new volume:
  // the input normalization and the sigmoid first (POST
  // /api/finetune/rank/input-norm, when they are not the session's as they
  // stand), then the threshold, which was fitted under them, and the
  // Channel: made the row's channel when `explicit` (Use these settings, a
  // click on that row), else filled only when none is picked (Fit to
  // Paint's). Says what it set on the line under Label it's buttons;
  // whether it could.
  async function applyRow(name, row, explicit) {
    if (changesInput(row) || changesOutput(row)) {
      const set = await postAnswer("/api/finetune/rank/input-norm", { input_norm: row.input_norm, ...chainParts(row) });
      if (!set.success) {
        sayFit(`Could not set the input normalization that fits ${name} to your paint: ${set.error}`, "danger");
        return false;
      }
      // The Processing tab's input normalization shows it as it now runs
      // (main.js reads the chain again), so a later Apply keeps it.
      announce(CHAIN_CHANGED_EVENT);
    }
    setFitted(name, row, explicit);
    return true;
  }

  // The threshold and the Channel `row` (model `name`'s score) fitted, set
  // and said on the line under Label it's buttons: the threshold as Seed
  // from Prediction's, and the Channel made the row's channel when
  // `explicit` (Use this model, Use these settings), else filled only when
  // none is picked (Fit to Paint's). Returns the channel set, by name and
  // number, or null.
  function setFitted(name, row, explicit) {
    installThreshold(name, row.threshold);
    let channel = null;
    if (Number.isInteger(row.channel_index)) {
      if (explicit) pickChannel(name, row.channel_index);
      if (explicit || fillChannel(name, row.channel_index)) channel = channelLabel(row.channel, row.channel_index);
    }
    sayFit(fittedText(name, row, channel), "success");
    return channel;
  }

  // Makes `candidate` the model to finetune, as its score (`row`) found it
  // best: the input normalization it was read through as the session's
  // (before the model is served, so it is served through it), and its
  // threshold as Seed from Prediction's, marked as set by ranking.
  // The ranking stays: the new volume holds the same paint, on the model's
  // grid.
  async function useModel(candidate, row) {
    if (running || switching !== null || fitting) return;
    if (!confirm(`Make ${candidate.name} the model to finetune? Your paint is copied to a new volume for it; the current volume is kept.`)) return;
    switching = candidate.name;
    rankBtn.disabled = true;
    showResults();
    say(`Making ${candidate.name} the model to finetune: serving it and copying your paint to a new volume...`);
    let d;
    try {
      // The channel too, so the model's layer opens on it rather than on
      // Neuroglancer's middle channel until the page's pick reaches it.
      d = await postAnswer("/api/finetune/rank/use", {
        ...request(candidate), ...chainParts(row),
        ...(Number.isInteger(row.channel_index) ? { select_channel: row.channel_index } : {}),
      });
    } catch (e) {
      d = { success: false, error: `Could not make ${candidate.name} the model to finetune: ${e}` };
    }
    switching = null;
    if (!d.success) {
      rankBtn.disabled = false;
      showResults();
      say(d.error || `Could not make ${candidate.name} the model to finetune.`, "danger");
      // The route may have done part of it before failing, or before its
      // answer was lost ("Failed to fetch"): the input normalization and
      // sigmoid set, the model started. Read the chain, the models and this
      // list again, so the Processing tab and the lists show what is so.
      announce(CHAIN_CHANGED_EVENT);
      announce(MODELS_CHANGED_EVENT);
      return;
    }
    training = { name: d.name, volumeId: d.volume_id };
    rankedOn = d.volume_id;
    // As the listing will have them, until it comes.
    for (const c of [...candidates, ...ranked.values()]) c.volume_model = c.name === d.name;
    const dropped = d.conflicts ? ` (${Number(d.conflicts).toLocaleString()} left unpainted where object and background met)` : "";
    // The Channel before the pick below (pickModel), which shows it.
    const channel = setFitted(d.name, row, true);
    // The Processing tab's input normalization shows it as it now runs
    // (main.js reads the chain again), so a later Apply keeps it.
    if (Array.isArray(d.input_norm) || Array.isArray(d.postprocess)) announce(CHAIN_CHANGED_EVENT);
    const set = [
      changesInput(row) && inputText(row),
      changesOutput(row) && outputText(row),
      Number.isFinite(row.threshold) && `Seed from Prediction's threshold is ${row.threshold}`,
      channel && channelSet(channel),
    ].filter(Boolean);
    summary = {
      text: `${d.name} is the model to finetune now, ${whyText(row)}: ${Number(d.painted_voxels || 0).toLocaleString()} painted voxels carried from ${d.from_volume} to ${d.volume_id}${dropped}.${set.length ? ` ${set.join("; ")}, as its score found best.` : ""}`,
      tone: "success",
    };
    rankBtn.disabled = false;
    showCandidates();
    showResults();
    say(summary.text, summary.tone);
    pickModel(d.name);
    sessionVolumeChanged(d.volume_id);
    // Its layer, and its server if it had none: the listing that follows
    // moves "(painted for)" to it too.
    announce(MODELS_CHANGED_EVENT);
  }

  // What names `candidate` to the routes: its name, how to build it, and
  // the GPU queues ticked at the top of the tab, where a model not running
  // starts (none before LSF lists them, nor in the browser: the routes then
  // use the Models tab's).
  function request(candidate) {
    const queue = form.gpuQueues();
    return { name: candidate.name, ...(candidate.entry ? { entry: candidate.entry } : {}), ...(queue.length ? { queue } : {}) };
  }

  // One model's score answer, with its name; a request that failed is an
  // answer that says so, for its row.
  function score(candidate) {
    return postAnswer("/api/finetune/rank/score", request(candidate))
      .then((d) => ({ ...d, name: candidate.name, error: d.success ? undefined : d.error || "no score" }))
      .catch((e) => ({ name: candidate.name, success: false, error: String(e) }));
  }

  async function rank() {
    if (running || switching !== null || fitting) return;
    running = true;
    setBusy(rankBtn, true, "Ranking...");
    // A volume made, or a model added, since the list was shown is in this one.
    const state = await listCandidates();
    // Every ticked model, which visible() keeps in view whatever its
    // section's search: what is ticked is scored, not only what is shown.
    const chosen = candidates.filter(isTicked);
    if (!state.ready || !chosen.length) {
      running = false;
      setBusy(rankBtn, false);
      showCandidates();
      if (state.ready) say("Tick at least one model to rank.", "warning");
      else showReadiness(state);
      return;
    }
    answers = [];
    summary = null;
    ranked = new Map(chosen.map((candidate) => [candidate.name, candidate]));
    rankedOn = state.volume.id;
    stopping = false;
    stopBtn.hidden = false;
    stopBtn.disabled = false;
    showCandidates();
    showResults();
    const started = performance.now();
    // The model the training form's Channel is shown for (the one painted
    // for, which has no Use this model) gets the channel that matched best,
    // when none is picked (fillChannel).
    let filled = null;
    for (let i = 0; i < chosen.length && !stopping; i++) {
      const starting = chosen[i].cluster && !chosen[i].served ? ", starting it on the cluster first" : "";
      say(`Scoring ${chosen[i].name} (${i + 1} of ${chosen.length})${starting}...`);
      const answer = await score(chosen[i]);
      answers.push(answer);
      if (isScored(answer) && Number.isInteger(answer.channel_index) && fillChannel(answer.name, answer.channel_index)) {
        filled = ` ${answer.name}'s Channel is now ${channelLabel(answer.channel, answer.channel_index)}, the one that matched best.`;
      }
      showResults();
    }
    running = false;
    setBusy(rankBtn, false);
    stopBtn.hidden = true;
    showCandidates();
    showResults();
    const seconds = Math.round((performance.now() - started) / 1000);
    const done = answers.length < chosen.length
      ? `Stopped after ${answers.length} of ${chosen.length} models`
      : `Ranked ${plural(answers.length, "model")}`;
    const [best, ...rest] = rankResults(answers).rows;
    const tied = rest.filter((row) => row.tied).length;
    summary = best && best.best
      ? {
        text: `${done} in ${seconds} s: ${best.name} matches your paint best, ${percent(best.matches)}` +
              (tied ? `; ${tied} more ${tied === 1 ? "is" : "are"} about as good.` : ".") + (filled || ""),
        tone: "muted",
      }
      : { text: `${done} in ${seconds} s: none could be scored (each row says why).`, tone: "danger" };
    say(summary.text, summary.tone);
  }

  // Fit to Paint: the model seeding the paint scored on it, as a ranking
  // scores one, and what fits it best set as Use this model sets it, with
  // no new volume: the input normalization first (POST
  // /api/finetune/rank/input-norm, when it is not the session's as it
  // stands), then the threshold, which was fitted under it and so is set
  // only once it is.
  async function fitToPaint() {
    if (running || switching !== null || fitting) return;
    const name = seedModel.value;
    if (!name) {
      sayFit("No model to fit: serve one on the Models tab, and pick it under Prediction from.", "warning");
      return;
    }
    fitting = true;
    setBusy(fitBtn, true, "Fitting...");
    rankBtn.disabled = true;
    showResults();
    sayFit(`Scoring ${name} on your paint...`);
    try {
      const d = await score({ name });
      if (!isScored(d)) {
        sayFit(`Could not fit ${name} to your paint: ${d.error}`, "danger");
        // A new volume has no paint to fit to yet.
        if (d.needs_volume) offerNewVolume(`Could not fit ${name} to your paint: ${d.error}`,
          () => sayFit("The annotation volume is made: paint on it, then fit."));
        return;
      }
      await applyRow(name, d, false);
    } catch (e) {
      sayFit(`Could not fit ${name} to your paint: ${e.message || e}`, "danger");
    } finally {
      fitting = false;
      setBusy(fitBtn, false);
      rankBtn.disabled = !ready;
      showResults();
    }
  }

  rankBtn.addEventListener("click", rank);
  fitBtn.addEventListener("click", fitToPaint);
  seedThreshold.addEventListener("input", showMark);
  seedThreshold.addEventListener("change", showMark);
  showMark();
  tickedOnly.addEventListener("change", showCandidates);
  // Tick the models shown, the hidden ones keeping their ticks; or untick
  // every one.
  tickAll.addEventListener("click", () => {
    for (const candidate of shown()) unticked.delete(candidate.name);
    showCandidates();
  });
  tickNone.addEventListener("click", () => {
    for (const candidate of candidates) unticked.add(candidate.name);
    showCandidates();
  });
  stopBtn.addEventListener("click", () => {
    stopping = true;
    stopBtn.disabled = true;
    say(`${status.textContent} Stopping after this one.`);
  });

  // Listed while the card is open: when the Training tab is shown or the
  // card opened there, and on Rank. While Rank waits for paint, also as the
  // pointer comes onto the card: the paint is made in the viewer beside it.
  // Once listed, again whenever the models served may have changed, for the
  // Show buttons, and when a model comes into the tab unserved (the browser
  // mode's Kept in this browser). A closed card lists nothing.
  const trainingPane = document.getElementById("training");
  document.getElementById("training-tab").addEventListener("shown.bs.tab", () => {
    if (section.open) refresh();
  });
  document.getElementById("rankCard").addEventListener("pointerenter", () => {
    if (section.open && rankBtn.disabled && !running && switching === null && !fitting) refresh();
  });
  const listAgain = () => {
    if (listed && section.open) refresh();
  };
  listen(MODELS_CHANGED_EVENT, listAgain);
  listen(MODEL_LIST_CHANGED_EVENT, listAgain);

  const open = readJSON(OPEN_KEY, null);
  if (typeof open === "boolean") section.open = open;
  section.addEventListener("toggle", () => {
    writeJSON(OPEN_KEY, section.open);
    if (section.open && trainingPane.classList.contains("active")) refresh();
  });
}
