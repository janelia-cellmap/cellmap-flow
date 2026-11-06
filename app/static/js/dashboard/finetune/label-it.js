// Label it: three buttons act on the patch on screen at once
// (routes/finetune/view_labels.py): label it from a model's prediction or
// all background, or relabel its objects by connected component, and Undo
// takes back the last of those or of an accepted AI annotation. How a seed
// makes objects of the prediction (its method) is picked among those that
// fit the chosen model's output. The patch is a box centred where the
// viewer looks, or with Field of view what the viewer's panel shows, which
// the routes work out from the viewer's state and the size of its frame
// here, the one thing about the view only the page knows.
//
// The same in both modes: in the browser (web/) the page answers the same
// routes (web/src/backend/routes/view-labels.ts), its prediction the model
// served in the page and its Undo the one the AI annotation's Accept
// shares, so the AI panel has no Undo of its own.
import { setBusy } from "../../lib/dom.js";
import { poll } from "../../lib/poll.js";
import { readJSON, writeJSON } from "../../lib/stored.js";
import { getAnswer, offerNewVolume, postAnswer, reloadViewer } from "./requests.js";
import { pickedChannel } from "./training-form.js";

// log: the Annotation Crops panel's log.
export function initLabelIt({ log }) {
  // what: "seed the view", say, for the log; describe(d): the log line for an
  // answer that changed labels. A box too large to label without asking is
  // answered needs_confirmation, and sent again confirmed; the button stays
  // busy until that answer too.
  // The seed settings (box, field of view, method, threshold, smallest
  // object, connectivity, per slice), kept in this browser, and shown when
  // one is set so a seed never silently uses an old one. Connectivity and
  // per slice are Split Objects' too; the box is every label action's.
  const seedSettings = document.getElementById("seedSettings");
  const seedMethod = document.getElementById("seedMethod");
  const seedMethodHint = document.getElementById("seedMethodHint");
  const seedThreshold = document.getElementById("seedThreshold");
  const seedMinSize = document.getElementById("seedMinSize");
  const seedConnectivity = document.getElementById("seedConnectivity");
  const seedPerSlice = document.getElementById("seedPerSlice");
  const seedBox = document.getElementById("seedBox");
  const seedFieldOfView = document.getElementById("seedFieldOfView");
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
  const saved = readJSON(SETTINGS_KEY, {});
  if (saved.threshold !== undefined && saved.threshold !== "") seedThreshold.value = saved.threshold;
  if (saved.min_size !== undefined) seedMinSize.value = saved.min_size;
  if (saved.connectivity) seedConnectivity.value = saved.connectivity;
  seedPerSlice.checked = Boolean(saved.per_slice);
  if (saved.box) seedBox.value = saved.box;
  seedFieldOfView.checked = Boolean(saved.field_of_view);
  // The method last picked by hand, used whenever the chosen model offers it.
  let chosenMethod = saved.method || "";
  // One model output patch (z, y, x voxels), the box when none is typed, as
  // the sources' answer gives it; null until it has.
  let patchVoxels = null;
  // The box, in annotation voxels: "z, y, x" or one number. Blank: the
  // server's default, one output patch. Anything else is sent as typed, for
  // the server to refuse with a reason. With Field of view, the routes take
  // the box from the viewer's panel and the typed box's depth along the axis
  // it looks along (blank: the slice on screen), and need the size of the
  // viewer's frame: its panels are in Neuroglancer's own page, which in
  // server mode is of another origin and cannot be measured from here.
  function boxBody() {
    const text = seedBox.value.trim();
    const body = text ? { box_voxels: text.split(/[\s,x×]+/).filter(Boolean).map(Number) } : {};
    if (seedFieldOfView.checked) {
      body.field_of_view = true;
      const frame = document.querySelector("#my_iframe");
      if (frame) body.viewer_px = [frame.clientWidth, frame.clientHeight];
    }
    return body;
  }
  // The box and the connected-components settings, shared by a seed and Split Objects.
  function componentsBody() {
    const body = boxBody();
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
  // With the training form's Channel when it is shown and picked: a binary
  // model of several channels then seeds from the channel the paint is for,
  // not from every channel.
  function seedRequest() {
    const body = seedBody();
    if (seedModel.value) body.model = seedModel.value;
    if (seedMethod.value) body.method = seedMethod.value;
    const channel = pickedChannel();
    if (channel !== null) body.select_channel = channel;
    return body;
  }
  function rememberSeedSettings() {
    writeJSON(SETTINGS_KEY, {
      threshold: seedThreshold.value, min_size: seedMinSize.value, method: chosenMethod,
      connectivity: seedConnectivity.value, per_slice: seedPerSlice.checked, box: seedBox.value.trim(),
      field_of_view: seedFieldOfView.checked,
    });
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
  for (const field of [seedBox, seedFieldOfView]) {
    field.addEventListener("change", () => {
      rememberSeedSettings();
      showBox();
    });
  }
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
  // One request at a time, and none while the page is hidden: asked every
  // 5 s regardless, they stack up behind a slow server until they hold
  // every connection the browser opens to the dashboard, and Apply and
  // everything else wait behind them.
  const undoViewLabelsBtn = document.getElementById("undoViewLabelsBtn");
  const seedModel = document.getElementById("seedModel");
  function showOnly(text) {
    seedModel.replaceChildren(new Option(text, ""));
    seedModel.disabled = true;
  }
  function refreshSeedSources({ signal, stale }) {
    return getAnswer("/api/finetune/view-labels/sources", { signal })
      .then((d) => {
        if (stale()) return;
        if (d && d.can_undo !== undefined) undoViewLabelsBtn.disabled = !d.can_undo;
        const names = (d && d.models) || [];
        methodsByModel = (d && d.methods) || {};
        patchVoxels = (d && d.box_voxels) || null;
        showBox();
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
      .catch(() => {
        if (!stale()) showOnly("could not list models");
      });
  }
  const seedSources = poll(refreshSeedSources, { intervalMs: 5000, pauseWhenHidden: true });
  // Opening the picker asks at once, cancelling a request still out.
  seedModel.addEventListener("mousedown", () => seedSources.restart());
  seedModel.addEventListener("change", showMethods);

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
          if (d.needs_volume) offerNewVolume(`Could not ${what}: ${d.error}`, () => labelView(button, url, what, describe, confirmed, body));
          else alert(`Could not ${what}:\n\n${d.error}`);
        } else if (d.reload_viewer) {
          log.add(describe(d, body) + (d.layer_refreshed ? "" : "; reloading the viewer"));
          if (!d.layer_refreshed) reloadViewer();
        } else {
          log.add(describe(d, body));
        }
        log.showEnd();
      })
      .catch((e) => log.add(`Could not ${what}: ${e}`))
      .finally(() => setBusy(button, false));
  }

  // The box an answer covered, for its log line: the field of view or the
  // box, and its size (the routes' shape_voxels).
  function boxCovered(d, body) {
    const size = d.shape_voxels ? ` (${d.shape_voxels.join(" × ")} voxels, z × y × x)` : "";
    return `${body.field_of_view ? "the field of view" : "the box"}${size}`;
  }

  function describeFill(d, body) {
    if (!d.reload_viewer) return `Nothing to label: every voxel of ${boxCovered(d, body)} is labelled already.`;
    const from = d.model ? ` from ${d.model}` : "";
    const by = d.method && METHODS[d.method] ? ` by ${METHODS[d.method].label.toLowerCase()}` : "";
    const how = d.threshold !== undefined && d.threshold !== null ? ` at threshold ${d.threshold}` : "";
    return `Labelled ${boxCovered(d, body)}${from}${by}${how}: ${d.filled_foreground} foreground and ${d.filled_background} background voxels`;
  }

  // A box one voxel thick (a field of view at the slice on screen) is split
  // in that slice alone, as each slice is with Per z slice: a wall there is
  // enough.
  function describeSplit(d, body) {
    const objects = `${d.objects} object${d.objects === 1 ? "" : "s"}`;
    const where = boxCovered(d, body);
    const oneSlice = seedPerSlice.checked || (d.shape_voxels || []).includes(1);
    if (!d.reload_viewer && oneSlice) return `Nothing to relabel in ${where}: ${objects}, each already one id.`;
    if (!d.reload_viewer) {
      return `Nothing to relabel in ${where}: ${objects}, each already one id. A cut has to go through ` +
             "every slice the object spans: paint the wall in the slices above and below too.";
    }
    return `Relabelled ${where}: ${objects}, ${d.split} split off, ${d.merged} merged`;
  }

  const seedViewBtn = document.getElementById("seedViewBtn");
  seedViewBtn.addEventListener("click", () =>
    labelView(seedViewBtn, "/api/finetune/view-labels/seed", "seed the view", describeFill, false, seedRequest()));
  const backgroundViewBtn = document.getElementById("backgroundViewBtn");
  backgroundViewBtn.addEventListener("click", () =>
    labelView(backgroundViewBtn, "/api/finetune/view-labels/background", "label the view background", describeFill,
              false, boxBody()));
  undoViewLabelsBtn.addEventListener("click", () =>
    labelView(undoViewLabelsBtn, "/api/finetune/view-labels/undo", "undo", (d) =>
      d.reload_viewer ? `Undid the last label action: ${d.restored} voxels restored`
                      : "Undid the last label action: every voxel it changed has been painted since"));
  const splitObjectsBtn = document.getElementById("splitObjectsBtn");
  splitObjectsBtn.addEventListener("click", () =>
    labelView(splitObjectsBtn, "/api/finetune/view-labels/split", "split the view's objects", describeSplit,
              false, componentsBody()));

  // The box the three buttons cover, said where it shows: the Box field's
  // placeholder (what blank means) and each button's title, the template's
  // with a sentence saying the box, so hovering a button says what it will
  // label.
  const boxButtons = [seedViewBtn, backgroundViewBtn, splitObjectsBtn];
  const boxButtonTitles = boxButtons.map((button) => button.title);
  function showBox() {
    const typed = seedBox.value.trim();
    let covers;
    if (seedFieldOfView.checked) {
      seedBox.placeholder = "one slice";
      const deep = typed ? "as deep as Box says along the axis it looks along" : "at the slice on screen";
      covers = `the field of view: what the viewer's panel shows (the top-left one in the four-panel layout), ${deep}`;
    } else {
      seedBox.placeholder = patchVoxels ? `${patchVoxels.join(", ")} (one patch)` : "one patch";
      const box = typed ? `${typed} voxels (z, y, x)`
        : `one model output patch${patchVoxels ? ` (${patchVoxels.join(" × ")} voxels, z × y × x)` : ""}`;
      covers = `${box}, centred where you look`;
    }
    boxButtons.forEach((button, i) => {
      button.title = `${boxButtonTitles[i]} It covers ${covers}.`;
    });
  }
  showBox();
}
