// Which model to start from (browser mode only): the models the browser can
// run, ranked by how well each one's prediction already matches the paint of
// the session's volume, so the one finetuned is the one with the least to
// learn. GET /api/finetune/rank/candidates lists them with the volume, and
// POST /api/finetune/rank/score scores one (web/src/backend/routes/rank.ts).
// The scores are asked for one at a time, as each can take up to a minute (a
// download, then a run over the painted chunks), and each is shown as it
// comes. A filter narrows the models listed; Rank scores the ticked ones it
// shows. Show serves a model and adds its layer, as the Models tab's Add
// does; Use this model (POST /api/finetune/rank/use) makes one the model to
// finetune: served, a new volume made for it with the paint carried onto its
// grid, that volume the session's and its paint layer in the viewer. The
// model picker (models.js pickModel) and the Annotation Crops panel's volume
// buttons (crops.js sessionVolumeChanged) follow.
import { h, setBusy } from "../../lib/dom.js";
import { announce, listen, MODELS_CHANGED_EVENT } from "../../lib/events.js";
import { readJSON, writeJSON } from "../../lib/stored.js";
import { sessionVolumeChanged } from "./crops.js";
import { pickModel } from "./models.js";
import { getAnswer, postAnswer } from "./requests.js";

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

// Whether `candidate` shows under the filter typed: every word of it is in
// the model's name, task, channels or a voxel size ("16 nm" or "16nm"), or
// is its "2d" or "3d", or "finetuned", whatever the case. Blank shows all.
export function matchesFilter(candidate, query) {
  const words = query.toLowerCase().replace(/(\d)\s+nm\b/g, "$1nm").split(/[\s,]+/).filter(Boolean);
  const text = [
    candidate.name, candidate.task, ...(candidate.channels || []),
    ...(candidate.voxel_size_nm || []).map((size) => `${size}nm`),
    candidate.dims && `${candidate.dims}d`, candidate.finetuned && "finetuned",
  ].filter(Boolean).join(" ").toLowerCase();
  return words.every((word) => text.includes(word));
}

function isScored(answer) {
  return Boolean(answer.success) && Number.isFinite(answer.matches);
}

// What a score is at besides the model as it is (web/src/finetune/rank.ts):
// the raw stretched from its window, when that scored clearly better, and the
// cut-off that fits the paint best, when it is not the model's own 0; with
// the score without the change.
function scoreNote(row) {
  if (Array.isArray(row.window)) {
    const cutoff = row.cutoff ? `, cut-off ${row.cutoff}` : "";
    return ` with the raw stretched from ${row.window.map((v) => Math.round(v)).join("–")}${cutoff} (${percent(row.matches_without_window)} without)`;
  }
  return Number.isFinite(row.cutoff) && row.cutoff !== 0 ? ` at cut-off ${row.cutoff} (${percent(row.matches_at_zero)} at 0)` : "";
}

// A cut-off (a logit) as Seed from Prediction's threshold, a probability.
function probability(cutoff) {
  return Math.round(1000 / (1 + Math.exp(-cutoff))) / 1000;
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

// Whether the card is open, kept in this browser as the patch tools'
// sections are (index.js rememberOpenSections).
const OPEN_KEY = "finetuneRankOpen";

export function initRank() {
  const section = document.getElementById("rankSection");
  const filter = document.getElementById("rankFilter");
  const tickAll = document.getElementById("rankTickAll");
  const tickNone = document.getElementById("rankTickNone");
  const candidateList = document.getElementById("rankCandidates");
  const rankBtn = document.getElementById("rankBtn");
  const stopBtn = document.getElementById("rankStopBtn");
  const status = document.getElementById("rankStatus");
  const results = document.getElementById("rankResults");
  const rows = document.getElementById("rankRows");
  const littlePaint = document.getElementById("rankLittlePaint");

  let candidates = [];
  // The models unticked by hand: one listed later starts ticked.
  const unticked = new Set();
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

  function say(text, tone = "muted") {
    status.textContent = text;
    status.className = `d-block text-${tone}`;
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
    return { ready: true, volume, message: `${plural(volume.painted_chunks, "painted chunk")} in ${volume.id}, painted for ${volume.model_name}.`, tone: "muted" };
  }

  // The status line and Rank from a listing; a ranking running, or one
  // shown, keeps its own line.
  function showReadiness(state) {
    if (running) return;
    rankBtn.disabled = !state.ready;
    if (state.ready && answers.length && summary) say(summary.text, summary.tone);
    else say(state.message, state.tone);
  }

  // A listing for the status line, one at a time.
  let refreshing = null;
  function refresh() {
    refreshing = refreshing || listCandidates().then(showReadiness).finally(() => { refreshing = null; });
    return refreshing;
  }

  // The models the filter shows.
  function shown() {
    return candidates.filter((candidate) => matchesFilter(candidate, filter.value));
  }

  function showCandidates() {
    tickAll.disabled = tickNone.disabled = running;
    const none = candidates.length && !shown().length
      && h("small", { class: "text-muted" }, "No model matches the filter.");
    candidateList.replaceChildren(none || "", ...candidates.map((candidate, i) => {
      const id = `rankCandidate${i}`;
      return h("div", { class: "form-check", hidden: !matchesFilter(candidate, filter.value) },
        h("input", {
          type: "checkbox", class: "form-check-input", id, value: candidate.name,
          checked: !unticked.has(candidate.name), disabled: running,
          onChange: (e) => (e.target.checked ? unticked.delete(candidate.name) : unticked.add(candidate.name)),
        }),
        h("label", { class: "form-check-label", for: id, title: `${candidate.repo}/${candidate.folder}` },
          candidate.name, candidate.volume_model ? " (painted for)" : "", " ",
          h("small", { class: "text-muted" }, describe(candidate))));
    }));
  }

  function showResults() {
    const { rows: table, littlePaint: little } = rankResults(answers);
    results.hidden = !table.length;
    littlePaint.hidden = !little;
    rows.replaceChildren(...table.map(resultRow));
  }

  // A model's row: its name and badges beside Use this model (a scored model
  // the volume is not for) and Show, its scores on a line under them, then
  // what they were scored on.
  function resultRow(row) {
    const candidate = ranked.get(row.name);
    const scored = isScored(row);
    const details = [
      row.channel && `${row.channel} channel`,
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
          candidate && scored && !candidate.volume_model && useButton(candidate, row),
          candidate && showButton(candidate))),
      scored
        ? h("div", {}, h("strong", {}, `${percent(row.matches)} match`),
          scoreNote(row),
          ` · objects ${percent(row.objects_found)} · background ${percent(row.background_right)}`)
        : h("div", { class: "small text-danger" }, row.error || "No score in the answer"),
      details && h("small", { class: "d-block text-muted" }, details));
  }

  function showButton(candidate) {
    const button = h("button", {
      type: "button", class: "btn btn-outline-secondary btn-sm", disabled: candidate.served,
      title: candidate.served ? "Served already: its layer is in the viewer" : "Serve this model and add its layer to the viewer",
    }, candidate.served ? "Shown" : "Show");
    button.addEventListener("click", () => showModel(candidate, button));
    return button;
  }

  function showModel(candidate, button) {
    setBusy(button, true);
    postAnswer("/api/models/add", { entry: { name: candidate.name, repo: candidate.repo, folder: candidate.folder } })
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
      type: "button", class: `btn btn-sm ${row.best ? "btn-primary" : "btn-outline-secondary"}`, disabled: running || switching !== null,
      title: "Finetune this model instead: your paint is copied to a new volume made for it, which becomes the session's, and the current volume is kept",
    }, "Use this model");
    if (switching === candidate.name) setBusy(button, true, "Copying paint...");
    button.addEventListener("click", () => useModel(candidate, row));
    return button;
  }

  // Makes `candidate` the model to finetune, as its score (`row`) found it
  // best: the raw's window, if one was chosen, as the session's normalizer,
  // and its cut-off as Seed from Prediction's threshold. The ranking stays:
  // the new volume holds the same paint, on the model's grid.
  async function useModel(candidate, row) {
    if (running || switching !== null) return;
    if (!confirm(`Make ${candidate.name} the model to finetune? Your paint is copied to a new volume for it; the current volume is kept.`)) return;
    switching = candidate.name;
    rankBtn.disabled = true;
    showResults();
    say(`Making ${candidate.name} the model to finetune: serving it and copying your paint to a new volume...`);
    let d;
    try {
      d = await postAnswer("/api/finetune/rank/use", { name: candidate.name, ...(Array.isArray(row.window) ? { window: row.window } : {}) });
    } catch (e) {
      d = { success: false, error: `Could not make ${candidate.name} the model to finetune: ${e}` };
    }
    switching = null;
    if (!d.success) {
      rankBtn.disabled = false;
      showResults();
      say(d.error || `Could not make ${candidate.name} the model to finetune.`, "danger");
      return;
    }
    training = { name: d.name, volumeId: d.volume_id };
    rankedOn = d.volume_id;
    // As the listing will have them, until it comes.
    for (const c of [...candidates, ...ranked.values()]) c.volume_model = c.name === d.name;
    const dropped = d.conflicts ? ` (${Number(d.conflicts).toLocaleString()} left unpainted where object and background met)` : "";
    const seedThreshold = document.getElementById("seedThreshold");
    const threshold = seedThreshold && Number.isFinite(row.cutoff) && row.cutoff !== 0 ? probability(row.cutoff) : null;
    if (threshold !== null) {
      seedThreshold.value = threshold;
      seedThreshold.dispatchEvent(new Event("change"));
    }
    const set = [
      Array.isArray(d.window) && `the input normalization stretches the raw from ${d.window.map((v) => Math.round(v)).join("–")}`,
      threshold !== null && `Seed from Prediction's threshold is ${threshold}`,
    ].filter(Boolean);
    summary = {
      text: `${d.name} is the model to finetune now: ${Number(d.painted_voxels || 0).toLocaleString()} painted voxels carried from ${d.from_volume} to ${d.volume_id}${dropped}.${set.length ? ` ${set.join("; ")}, as its score found best.` : ""}`,
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

  // One model's score answer, with its name; a request that failed is an
  // answer that says so, for its row.
  function score(candidate) {
    return postAnswer("/api/finetune/rank/score", { name: candidate.name })
      .then((d) => ({ ...d, name: candidate.name, error: d.success ? undefined : d.error || "no score" }))
      .catch((e) => ({ name: candidate.name, success: false, error: String(e) }));
  }

  async function rank() {
    if (running || switching !== null) return;
    running = true;
    setBusy(rankBtn, true, "Ranking...");
    // A volume made, or a model added, since the list was shown is in this one.
    const state = await listCandidates();
    const chosen = shown().filter((candidate) => !unticked.has(candidate.name));
    if (!state.ready || !chosen.length) {
      running = false;
      setBusy(rankBtn, false);
      showCandidates();
      if (state.ready) say("Tick at least one of the models shown to rank.", "warning");
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
    for (let i = 0; i < chosen.length && !stopping; i++) {
      say(`Scoring ${chosen[i].name} (${i + 1} of ${chosen.length})...`);
      answers.push(await score(chosen[i]));
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
              (tied ? `; ${tied} more ${tied === 1 ? "is" : "are"} about as good.` : "."),
        tone: "muted",
      }
      : { text: `${done} in ${seconds} s: none could be scored (each row says why).`, tone: "danger" };
    say(summary.text, summary.tone);
  }

  rankBtn.addEventListener("click", rank);
  filter.addEventListener("input", showCandidates);
  // Tick or untick the models shown; the hidden ones keep their ticks.
  tickAll.addEventListener("click", () => {
    for (const candidate of shown()) unticked.delete(candidate.name);
    showCandidates();
  });
  tickNone.addEventListener("click", () => {
    for (const candidate of shown()) unticked.add(candidate.name);
    showCandidates();
  });
  stopBtn.addEventListener("click", () => {
    stopping = true;
    stopBtn.disabled = true;
    say(`${status.textContent} Stopping after this one.`);
  });

  // Listed when the Training tab is opened, and on Rank. While Rank waits
  // for paint, also as the pointer comes onto the card: the paint is made in
  // the viewer beside it. Once listed, again whenever the models served may
  // have changed, for the Show buttons.
  document.getElementById("training-tab").addEventListener("shown.bs.tab", refresh);
  document.getElementById("rankCard").addEventListener("pointerenter", () => {
    if (rankBtn.disabled && !running && switching === null) refresh();
  });
  listen(MODELS_CHANGED_EVENT, () => {
    if (listed) refresh();
  });

  const open = readJSON(OPEN_KEY, null);
  if (typeof open === "boolean") section.open = open;
  section.addEventListener("toggle", () => writeJSON(OPEN_KEY, section.open));
}
