// Good regions: views the user marks as ones the model already gets right.
// Training rehearses them (holds the model to what it predicts there), and
// the rehearsal setting's hint says how much that will weigh. Beside them,
// three buttons act on the same patch at once (routes/finetune/view_labels.py):
// label it from the model's prediction or all background, or relabel its
// objects by connected component. How a seed makes objects of the prediction
// (its method) is picked among those that fit the chosen model's output.
import { setBusy } from "../../lib/dom.js";
import { getAnswer, postAnswer } from "./requests.js";

// log: the Annotation Crops panel's log.
export function initGoodRegions({ log }) {
  const markGoodRegionBtn = document.getElementById("markGoodRegionBtn");
  const goodRegionCount = document.getElementById("goodRegionCount");
  const rehearsalFraction = document.getElementById("rehearsalFraction");

  // Last count seen, so the rehearsal hint can say whether the setting will
  // do anything without re-fetching every time the dropdown moves.
  let lastGoodRegionCount = 0;

  function renderGoodRegionCount(count) {
    lastGoodRegionCount = count || 0;
    goodRegionCount.textContent = count
      ? `${count} good region${count === 1 ? "" : "s"}`
      : "none yet";
    updateRehearsalHint();
  }

  function updateRehearsalHint() {
    const hint = document.getElementById("rehearsalFractionHint");

    // The label's tooltip says what the setting is; this says what it does now.
    if (lastGoodRegionCount === 0) {
      hint.textContent = "No good regions marked: no effect.";
      return;
    }
    const raw = rehearsalFraction.value.trim();
    // Blank means the manifest decides, which is 25% unless it says otherwise.
    const fraction = raw === "" ? 0.25 : Number(raw);
    if (!Number.isFinite(fraction) || fraction <= 0) {
      hint.textContent =
        `0: the ${lastGoodRegionCount} good region` +
        `${lastGoodRegionCount === 1 ? " is" : "s are"} ignored.`;
      return;
    }
    hint.textContent =
      `~${Math.round(fraction * 100)}% of patches from ` +
      `${lastGoodRegionCount} good region${lastGoodRegionCount === 1 ? "" : "s"}.`;
  }

  markGoodRegionBtn.addEventListener("click", function () {
    setBusy(markGoodRegionBtn, true);
    postAnswer("/api/finetune/good-regions/mark-view", {})
      .then((d) => {
        if (d.success) {
          renderGoodRegionCount(d.count);
          const at = d.region.offset_nm.map((v) => Math.round(v)).join(", ");
          log.add(`Marked ${d.region.label} as good at [${at}] nm (${d.count} total)`);
        } else {
          // A mark that could not be saved must be loud: it would otherwise
          // look drawn, and be gone.
          log.add(`Could not mark region: ${d.error}`);
          alert(`Could not mark region as good:\n\n${d.error}`);
        }
        log.showEnd();
      })
      .catch((e) => {
        log.add(`Could not mark region: ${e}`);
      })
      .finally(() => setBusy(markGoodRegionBtn, false));
  });

  document.getElementById("clearGoodRegionsBtn").addEventListener("click", function () {
    // Deletes every region in the session, and there is no undo; the button
    // sits right next to "Mark This View as Good".
    const what = lastGoodRegionCount
      ? `all ${lastGoodRegionCount} good region${lastGoodRegionCount === 1 ? "" : "s"}`
      : "all good regions";
    if (!confirm(`Remove ${what} from this session? This cannot be undone.`)) return;
    postAnswer("/api/finetune/good-regions/delete", {})
      .then((d) => {
        renderGoodRegionCount(d.count || 0);
        log.add("Cleared good regions");
      })
      .catch(() => {});
  });

  // Neuroglancer keeps the chunks it has read. The server re-reads the paint
  // layer alone (answer's layer_refreshed); only when it could not does the
  // whole viewer reload, getting its state back from the dashboard. Absent
  // when no viewer is connected.
  function reloadViewer() {
    const frame = document.querySelector("#my_iframe");
    if (frame) frame.src = frame.src;
  }

  // what: "seed the view", say, for the log; describe(d): the log line for an
  // answer that changed labels. A box too large to label without asking is
  // answered needs_confirmation, and sent again confirmed; the button stays
  // busy until that answer too.
  // The seed settings (method, threshold, smallest object, connectivity,
  // per slice), kept in this browser, and shown when one is set so a seed
  // never silently uses an old one. Connectivity and per slice are Split
  // Objects' too.
  const seedSettings = document.getElementById("seedSettings");
  const seedMethod = document.getElementById("seedMethod");
  const seedMethodHint = document.getElementById("seedMethodHint");
  const seedThreshold = document.getElementById("seedThreshold");
  const seedMinSize = document.getElementById("seedMinSize");
  const seedConnectivity = document.getElementById("seedConnectivity");
  const seedPerSlice = document.getElementById("seedPerSlice");
  const SETTINGS_KEY = "cellmap_flow.seed_settings";
  // The seed methods (view_labels.SEED_METHODS): their name in the picker,
  // a line on what they do, and which settings they read.
  const METHODS = {
    instances: {
      label: "Model's instances",
      hint: "The ids the model serves (Cellpose's masks), one object each; touching objects stay apart.",
      needs: "a model serving instance ids, e.g. Cellpose with Output: Masks",
      uses: ["min_size", "connectivity"],
    },
    mutex_watershed: {
      label: "Mutex watershed",
      hint: "Affinities: neighbours join where the model's affinity is over the threshold, so touching objects split.",
      needs: "an affinity model",
      uses: ["threshold", "min_size"],
    },
    distance_watershed: {
      label: "Distance watershed",
      hint: "Distance: objects grow from the distance's peaks, so touching objects split where it dips between them.",
      needs: "a distance model",
      uses: ["threshold", "min_size", "connectivity"],
    },
    components: {
      label: "Threshold + components",
      hint: "Over the threshold is foreground, each connected object an id; touching objects stay one.",
      uses: ["threshold", "min_size", "connectivity", "per_slice"],
    },
  };
  // The method last picked by hand, used whenever the chosen model offers it.
  let chosenMethod = "";
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    if (saved.threshold !== undefined && saved.threshold !== "") seedThreshold.value = saved.threshold;
    if (saved.min_size !== undefined) seedMinSize.value = saved.min_size;
    if (saved.connectivity) seedConnectivity.value = saved.connectivity;
    seedPerSlice.checked = Boolean(saved.per_slice);
    chosenMethod = saved.method || "";
  } catch (e) { /* no storage: defaults */ }
  // The connected-components settings, shared by a seed and Split Objects.
  function componentsBody() {
    const body = {};
    if (seedConnectivity.value !== "1") body.connectivity = Number(seedConnectivity.value);
    if (seedPerSlice.checked) body.per_slice = true;
    return body;
  }
  function seedBody() {
    const body = componentsBody();
    if (seedThreshold.value !== "") body.threshold = Number(seedThreshold.value);
    if (Number(seedMinSize.value) > 0) body.min_size = Number(seedMinSize.value);
    return body;
  }
  function seedRequest() {
    const body = seedBody();
    if (seedModel.value) body.model = seedModel.value;
    if (seedMethod.value) body.method = seedMethod.value;
    return body;
  }
  function rememberSeedSettings() {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify({
        threshold: seedThreshold.value, min_size: seedMinSize.value, method: chosenMethod,
        connectivity: seedConnectivity.value, per_slice: seedPerSlice.checked,
      }));
    } catch (e) { /* no storage */ }
  }
  // The method's line, and the settings it does not read greyed out (Split
  // Objects reads connectivity and per slice whatever the method).
  function showMethod() {
    const method = METHODS[seedMethod.value];
    seedMethodHint.textContent = method ? method.hint : "";
    const uses = method ? method.uses : [];
    seedThreshold.disabled = !uses.includes("threshold");
    seedMinSize.disabled = !uses.includes("min_size");
  }
  seedThreshold.addEventListener("change", rememberSeedSettings);
  seedMinSize.addEventListener("change", rememberSeedSettings);
  seedConnectivity.addEventListener("change", rememberSeedSettings);
  seedPerSlice.addEventListener("change", rememberSeedSettings);
  seedMethod.addEventListener("change", () => {
    chosenMethod = seedMethod.value;
    rememberSeedSettings();
    showMethod();
  });
  // Open whenever a setting is set, so a seed never silently uses an old one.
  seedSettings.open = Object.keys(seedBody()).length > 0;

  // The methods the chosen model offers (the sources' answer), best fit
  // first: the one picked by hand when it is among them, else the first.
  let methodsByModel = {};
  // Every method is listed, the chosen model's in its best-fit order and the
  // rest after them greyed out with what they need: hiding them left no way
  // to tell that the others exist, or what would make them available.
  function showMethods() {
    const offered = methodsByModel[seedModel.value] || ["components"];
    const order = offered.concat(Object.keys(METHODS).filter((m) => !offered.includes(m)));
    const options = order.map((m) => {
      const method = METHODS[m] || { label: m };
      const fits = offered.includes(m);
      const option = new Option(fits ? method.label : `${method.label} (needs ${method.needs || "another model"})`, m);
      option.disabled = !fits;
      return option;
    });
    const key = (opts) => opts.map((o) => `${o.value}|${o.disabled}`).join(",");
    if (key(Array.from(seedMethod.options)) !== key(options)) seedMethod.replaceChildren(...options);
    seedMethod.value = offered.includes(chosenMethod) ? chosenMethod : offered[0];
    showMethod();
  }

  // The models a seed can read: every running server, newest first, the
  // default (the volume model's latest finetune, else that model) marked.
  // Refreshed every few seconds and before the picker opens, since models
  // and finetunes come and go; the choice is kept while it is still running.
  const undoViewLabelsBtn = document.getElementById("undoViewLabelsBtn");
  const seedModel = document.getElementById("seedModel");
  function showOnly(text) {
    seedModel.replaceChildren(new Option(text, ""));
    seedModel.disabled = true;
  }
  function refreshSeedSources() {
    return getAnswer("/api/finetune/view-labels/sources")
      .then((d) => {
        if (d && d.can_undo !== undefined) undoViewLabelsBtn.disabled = !d.can_undo;
        const names = (d && d.models) || [];
        methodsByModel = (d && d.methods) || {};
        if (!names.length) {
          showOnly("no model running");
          return showMethods();
        }
        const chosen = seedModel.value;
        const options = names.slice().reverse().map((name) =>
          new Option(name === d.default ? `${name} (default)` : name, name));
        // Rebuilt only when the list changed, so an open picker is not reset under the cursor.
        const now = Array.from(seedModel.options).map((o) => `${o.value}|${o.text}`).join(",");
        if (now !== options.map((o) => `${o.value}|${o.text}`).join(",")) seedModel.replaceChildren(...options);
        seedModel.disabled = false;
        seedModel.value = names.includes(chosen) ? chosen : (d.default || names[names.length - 1]);
        showMethods();
      })
      .catch(() => showOnly("could not list models"));
  }
  seedModel.addEventListener("mousedown", refreshSeedSources);
  seedModel.addEventListener("change", showMethods);
  refreshSeedSources();
  setInterval(refreshSeedSources, 5000);

  function labelView(button, url, what, describe, confirmed, body = {}) {
    setBusy(button, true);
    return postAnswer(url, confirmed ? { ...body, confirm: true } : body)
      .then((d) => {
        if (d.needs_confirmation && !confirmed) {
          return confirm(d.error) ? labelView(button, url, what, describe, true, body) : undefined;
        }
        if (d.can_undo !== undefined) undoViewLabelsBtn.disabled = !d.can_undo;
        if (!d.success) {
          log.add(`Could not ${what}: ${d.error}`);
          alert(`Could not ${what}:\n\n${d.error}`);
        } else if (d.reload_viewer) {
          log.add(describe(d) + (d.layer_refreshed ? "" : "; reloading the viewer"));
          if (!d.layer_refreshed) reloadViewer();
        } else {
          log.add(describe(d));
        }
        log.showEnd();
      })
      .catch((e) => log.add(`Could not ${what}: ${e}`))
      .finally(() => setBusy(button, false));
  }

  function describeFill(d) {
    if (!d.reload_viewer) return "Nothing to label: every voxel of the view is labelled already.";
    const from = d.model ? ` from ${d.model}` : "";
    const by = d.method && METHODS[d.method] ? ` by ${METHODS[d.method].label.toLowerCase()}` : "";
    const how = d.threshold !== undefined && d.threshold !== null ? ` at threshold ${d.threshold}` : "";
    return `Labelled the view${from}${by}${how}: ${d.filled_foreground} foreground and ${d.filled_background} background voxels`;
  }

  function describeSplit(d) {
    const objects = `${d.objects} object${d.objects === 1 ? "" : "s"}`;
    if (!d.reload_viewer && seedPerSlice.checked) return `Nothing to relabel: ${objects}, each already one id.`;
    if (!d.reload_viewer) {
      return `Nothing to relabel: ${objects}, each already one id. A cut has to go through ` +
             "every slice the object spans: paint the wall in the slices above and below too.";
    }
    return `Relabelled the view: ${objects}, ${d.split} split off, ${d.merged} merged`;
  }

  const seedViewBtn = document.getElementById("seedViewBtn");
  seedViewBtn.addEventListener("click", () =>
    labelView(seedViewBtn, "/api/finetune/view-labels/seed", "seed the view", describeFill, false, seedRequest()));
  const backgroundViewBtn = document.getElementById("backgroundViewBtn");
  backgroundViewBtn.addEventListener("click", () =>
    labelView(backgroundViewBtn, "/api/finetune/view-labels/background", "label the view background", describeFill));
  undoViewLabelsBtn.addEventListener("click", () =>
    labelView(undoViewLabelsBtn, "/api/finetune/view-labels/undo", "undo", (d) =>
      d.reload_viewer ? `Undid the last label action: ${d.restored} voxels restored`
                      : "Undid the last label action: every voxel it changed has been painted since"));
  const splitObjectsBtn = document.getElementById("splitObjectsBtn");
  splitObjectsBtn.addEventListener("click", () =>
    labelView(splitObjectsBtn, "/api/finetune/view-labels/split", "split the view's objects", describeSplit,
              false, componentsBody()));

  rehearsalFraction.addEventListener("change", updateRehearsalHint);

  getAnswer("/api/finetune/good-regions")
    .then((d) => renderGoodRegionCount(d.count || 0))
    .catch(() => renderGoodRegionCount(0));
}
