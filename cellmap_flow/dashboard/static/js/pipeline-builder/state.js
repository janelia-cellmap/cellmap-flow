// The pipeline the builder edits, and what keeps the server in step with it.
//
// `pipeline` holds one list of nodes per node type, and the edges:
//   inputs, outputs, normalizers, models, postprocessors, blockwise_config
//   (the blockwise job settings), edges ({id, from, to}).
// A node is {id, name, params, position}. A model node may also carry the
// config it was defined with (its ModelConfig.to_dict()), which the blockwise
// routes read. The keys and shapes are what /api/blockwise/* take.
//
// Whatever edits the pipeline calls edited(): the change is applied to the
// server (PUT /api/pipeline, 2 s after the last edit, which also redraws the
// viewer's layers) and the blockwise steps, which ran on the pipeline as it
// was, start over. Leaving the page with a change the server has not had
// sends it in a request that outlives the page.
import { postJSON } from "../lib/api.js";
import { pageData } from "../lib/page-data.js";
import { showMessage } from "./messages.js";

const PAGE = pageData();
const asList = (items) => (Array.isArray(items) ? items : Object.values(items));

// The dataset the dashboard was started on; an INPUT node without a path of
// its own gets it.
export const datasetPath = PAGE.dataset_path || "";

// What the palette offers, by node type.
export const palette = {
  normalizer: asList(PAGE.input_normalizers),
  model: asList(PAGE.available_models),
  postprocessor: asList(PAGE.output_postprocessors),
};

// The settings a blockwise-config node holds, as the blockwise job takes them.
export const BLOCKWISE_FIELDS = [
  { key: "queue", label: "Queue" },
  { key: "charge_group", label: "Charge Group" },
  { key: "nb_cores_master", label: "Cores Master" },
  { key: "nb_cores_worker", label: "Cores Worker" },
  { key: "nb_workers", label: "Workers" },
  { key: "tmp_dir", label: "Temp Directory" },
  { key: "blockwise_tasks_dir", label: "Tasks Directory" },
];

// Those settings picked out of an object that has them (and maybe more).
export function blockwiseSettings(from) {
  return Object.fromEntries(BLOCKWISE_FIELDS.map(({ key }) => [key, from[key]]));
}

export const pipeline = {
  inputs: [], outputs: [], normalizers: [], models: [], postprocessors: [], blockwise_config: [], edges: [],
};

// Each node type's list in `pipeline`, in the order the canvas draws them.
const LISTS = {
  input: "inputs",
  output: "outputs",
  normalizer: "normalizers",
  model: "models",
  postprocessor: "postprocessors",
  "blockwise-config": "blockwise_config",
};
export const NODE_TYPES = Object.keys(LISTS);

export function nodesOf(type) {
  return Object.hasOwn(LISTS, type) ? pipeline[LISTS[type]] : undefined;
}

export function findNode(type, id) {
  return (nodesOf(type) || []).find((n) => n.id === id);
}

// Put a whole new pipeline in place (an import): every list at once.
export function replacePipeline(lists) {
  Object.assign(pipeline, lists);
}

// A node of a type the builder does not know (a drop of something that is not
// a palette entry) goes in no list.
export function addNodeTo(type, node) {
  nodesOf(type)?.push(node);
}

// Take a node out, with every edge to or from it.
export function removeNodeFrom(type, id) {
  pipeline.edges = pipeline.edges.filter((e) => e.from !== id && e.to !== id);
  if (nodesOf(type)) pipeline[LISTS[type]] = nodesOf(type).filter((n) => n.id !== id);
}

// Where the i-th node of a type goes when nothing says otherwise: INPUT and
// OUTPUT nodes in a column each, the others in a row.
export function defaultPosition(type, i) {
  switch (type) {
    case "input": return { x: 20, y: 20 + i * 180 };
    case "output": return { x: 900, y: 20 + i * 180 };
    case "normalizer": return { x: 200 + i * 380, y: 20 };
    case "model": return { x: 400 + i * 380, y: 20 };
    case "postprocessor": return { x: 600 + i * 380, y: 20 };
    default: return { x: 100, y: 400 };  // blockwise-config
  }
}

// A pipeline's lists from a saved one: the page's starting pipeline or an
// imported file (io.js). Every node gets an id, and keeps the position it
// was saved with or gets its type's default; an op may be given by its name
// alone; a model keeps its config (see readModel). Returns {lists,
// unplaced}, unplaced being the ids of the nodes that came without a
// position.
export function readPipeline(data) {
  const unplaced = [];
  const node = (n, prefix, type, i, name, params) => {
    const id = n.id || `${prefix}-${i}-${Date.now()}`;
    if (!n.position) unplaced.push(id);
    return { id, name, params, position: n.position || defaultPosition(type, i) };
  };
  const io = (prefix, name, params) => (n, i) => node(n, prefix, prefix, i, name, n.params || params);
  const op = (prefix, type) => (n, i) => (typeof n === "string"
    ? node({}, prefix, type, i, n, {})
    : node(n, prefix, type, i, n.name, n.params || {}));
  const lists = {
    inputs: (data.inputs || []).map(io("input", "INPUT", { dataset_path: datasetPath })),
    outputs: (data.outputs || []).map(io("output", "OUTPUT", {})),
    normalizers: (data.input_normalizers || data.normalizers || []).map(op("norm", "normalizer")),
    models: (data.models || []).map((m, i) => (typeof m === "string"
      ? node({}, "model", "model", i, m, {})
      : readModel(m, node(m, "model", "model", i, m.name, m.params || {})))),
    postprocessors: (data.postprocessors || []).map(op("post", "postprocessor")),
    blockwise_config: (data.blockwise_config || []).map((c, i) => node(
      c, "blockwise", "blockwise-config", i, "Blockwise Configuration", c.params)),
    edges: data.edges || [],
  };
  return { lists, unplaced };
}

// The page's starting pipeline (pageData().pipeline): the live chain's
// steps as normalizer and postprocessor nodes, and the rest of what the
// builder last applied (routes/pipeline_builder_page.py), read as
// readPipeline reads it. There is always an INPUT node, with the
// dashboard's dataset if it has no path of its own.
//
// A node keeps the position it was saved with. Returns the ids of the nodes
// that had none, for the page to lay out once they are drawn
// (canvas.autoLayoutNodes): every node, before anything was applied; a step
// added since by something other than the builder (Submit on the dashboard
// page). The saved edges stay if they still join the chain as it is (the
// user's own extra edges with them); otherwise the chain changed since, a
// step added or gone, and they are rebuilt from the node order.
export function loadPipeline() {
  const { lists, unplaced } = readPipeline(PAGE.pipeline);
  if (lists.inputs.length === 0) {
    const input = { id: `input-0-${Date.now()}`, name: "INPUT", params: { dataset_path: datasetPath },
                    position: defaultPosition("input", 0) };
    lists.inputs.push(input);
    unplaced.push(input.id);
  } else if (!lists.inputs[0].params?.dataset_path) {
    lists.inputs[0].params = lists.inputs[0].params || {};
    lists.inputs[0].params.dataset_path = datasetPath;
  }
  replacePipeline(lists);
  const linked = new Set(pipeline.edges.map((e) => `${e.from}>${e.to}`));
  if (!chainEdges().every(([from, to]) => linked.has(`${from}>${to}`))) autoConnectNodes();
  return unplaced;
}

// A saved model node m, as `model` ({id, name, params, position}), with its
// config: m.config (the server's ModelConfig.to_dict()), or else, for a
// model written with its config's fields as its own (type, ...), those.
// The config's channel names become an array: 'channels' (FlyModel etc.)
// or 'channels_names' (HuggingFace), either of which may come as a JSON
// string. 'channels' is what the rest of the builder reads.
function readModel(m, model) {
  let config = m.config && typeof m.config === "object" ? m.config : null;
  if (!config && m.type) {
    config = { ...m };
    delete config.id;
    delete config.params;
    delete config.position;
  }
  if (config) {
    model.config = config;
    const chKey = model.config.channels ? "channels" : (model.config.channels_names ? "channels_names" : null);
    if (chKey) {
      if (typeof model.config[chKey] === "string") {
        try {
          model.config[chKey] = JSON.parse(model.config[chKey]);
          if (!Array.isArray(model.config[chKey])) {
            model.config[chKey] = [model.config[chKey]];
          }
        } catch {
          model.config[chKey] = [model.config[chKey]];
        }
      }
      if (chKey === "channels_names" && !model.config.channels) {
        model.config.channels = model.config.channels_names;
      }
    }
    // A node without params of its own shows its config's (all but 'name').
    if (!m.params || Object.keys(m.params).length === 0) {
      model.params = {};
      Object.entries(config).forEach(([key, value]) => {
        if (key !== "name") {
          model.params[key] = value;
        }
      });
    }
  }
  return model;
}

// An edge from one node to another, unless there is one already; true if it
// was added. An edge the user draws passes its own id (edge-<time>); the
// ones the builder makes get a random part too, as several are made at once.
export function addEdge(fromId, toId, id) {
  if (!fromId || !toId || pipeline.edges.some((e) => e.from === fromId && e.to === toId)) return false;
  pipeline.edges.push({ id: id || `edge-${Date.now()}-${Math.random()}`, from: fromId, to: toId });
  return true;
}

// The edges a pipeline in the usual order has, as [from id, to id]: INPUT
// -> the normalizers in a chain -> every model -> the postprocessors in a
// chain -> OUTPUT, skipping any stage that has no nodes.
function chainEdges() {
  const edges = [];
  const link = (from, to) => edges.push([from, to]);
  const { inputs, normalizers, models, postprocessors, outputs } = pipeline;

  if (inputs.length > 0 && normalizers.length > 0) {
    link(inputs[0].id, normalizers[0].id);
  }
  for (let i = 0; i < normalizers.length - 1; i++) {
    link(normalizers[i].id, normalizers[i + 1].id);
  }
  const sourceForModels = normalizers.length > 0
    ? normalizers[normalizers.length - 1].id
    : (inputs.length > 0 ? inputs[0].id : null);
  if (sourceForModels) {
    models.forEach((model) => link(sourceForModels, model.id));
  }
  if (postprocessors.length > 0) {
    models.forEach((model) => link(model.id, postprocessors[0].id));
  } else if (outputs.length > 0) {
    models.forEach((model) => link(model.id, outputs[0].id));
  }
  for (let i = 0; i < postprocessors.length - 1; i++) {
    link(postprocessors[i].id, postprocessors[i + 1].id);
  }
  if (postprocessors.length > 0 && outputs.length > 0) {
    link(postprocessors[postprocessors.length - 1].id, outputs[0].id);
  }
  return edges;
}

// Replace the edges with the usual order's (chainEdges).
export function autoConnectNodes() {
  pipeline.edges = [];
  chainEdges().forEach(([from, to]) => addEdge(from, to));
}

// Connect a node just added, without touching the other edges: an INPUT to
// the first normalizer (or model), an OUTPUT from the last postprocessor (or
// model), a model between the last normalizer (or INPUT) and the first
// postprocessor (or OUTPUT). Other nodes are connected by hand.
export function connectNewNode(node, type) {
  const { inputs, normalizers, models, postprocessors, outputs } = pipeline;
  if (type === "input") {
    const target = normalizers[0] || models[0];
    if (target) addEdge(node.id, target.id);
  } else if (type === "output") {
    const source = postprocessors.length > 0
      ? postprocessors[postprocessors.length - 1]
      : (models.length > 0 ? models[models.length - 1] : null);
    if (source) addEdge(source.id, node.id);
  } else if (type === "model") {
    const source = normalizers.length > 0
      ? normalizers[normalizers.length - 1]
      : (inputs.length > 0 ? inputs[0] : null);
    if (source) addEdge(source.id, node.id);
    const target = postprocessors[0] || (outputs.length > 0 ? outputs[0] : null);
    if (target) addEdge(node.id, target.id);
  }
}

// The parameters a new normalizer, model or postprocessor node starts with,
// from the palette this page was rendered with -- and so what the node's ↻
// button resets it to. Returns fresh copies ({params, config} with config
// only for a model that has one), or null when the palette does not list
// the name.
export function nodeDefaults(type, name) {
  const defs = palette[type];
  if (!defs) return null;
  const def = defs.find((d) => (typeof d === "string" ? d : d.name) === name);
  if (!def || typeof def !== "object") return null;
  const copy = (value) => JSON.parse(JSON.stringify(value));
  let config = null;
  if (type === "model") {
    if (def.config) {
      config = copy(def.config);
    } else if (def.type) {
      // A configured model's entry is its to_dict() itself, with no nested
      // config: the whole entry is the config.
      config = copy(def);
    }
  }
  if (!config) {
    return { params: def.params ? copy(def.params) : {} };
  }
  // The node shows its config's fields (all but 'name') as its params.
  const params = {};
  Object.entries(config).forEach(([key, value]) => {
    if (key !== "name") {
      params[key] = value;
    }
  });
  return { params, config };
}

// A model node's channel names as an array: config.channels (FlyModel etc.)
// or config.channels_names (HuggingFace), either of which may still be a
// JSON string after an import.
export function modelChannels(model) {
  const config = (model && model.config) || {};
  const channels = config.channels || config.channels_names;
  if (Array.isArray(channels)) return channels;
  if (typeof channels === "string") {
    try {
      const parsed = JSON.parse(channels);
      return Array.isArray(parsed) ? parsed : [channels];
    } catch {
      return [channels];
    }
  }
  return [];
}

// ---- keeping the server in step ------------------------------------------

const editListeners = [];

// The changes made so far (edits, moves), and how many of them the server
// had when an apply last succeeded: the page is "dirty" while they differ.
let changes = 0;
let appliedChanges = 0;

// A change that is not an edit of what the pipeline computes: nodes moved,
// by Auto Arrange or a drag (a drag also schedules an apply).
export function changed() {
  changes += 1;
}

// listener() runs after every edit of the pipeline.
export function onEdit(listener) {
  editListeners.push(listener);
}

// The pipeline changed. Unless `apply` is false the change is applied to the
// server; the edits that pass false (bounding boxes, the separate-zarrs box,
// an import) have never been applied on their own, and reach the server with
// the next apply or the unload beacon.
export function edited({ apply = true } = {}) {
  changed();
  if (apply) scheduleApply();
  editListeners.forEach((listener) => listener());
}

// The pipeline as PUT /api/pipeline takes it: the two chains, and the
// builder's canvas, which the server keeps for the page's next load. A step
// is its node's params and then its name, so the name wins over a param of
// that name. Model configs are left out.
function buildPipelineBody() {
  const step = (node) => ({ ...node.params, name: node.name });
  return {
    input_norm: pipeline.normalizers.map(step),
    postprocess: pipeline.postprocessors.map(step),
    builder: {
      inputs: pipeline.inputs.map((i) => ({ id: i.id, params: i.params, position: i.position })),
      outputs: pipeline.outputs.map((o) => ({ id: o.id, params: o.params, position: o.position })),
      edges: pipeline.edges.map((e) => ({ id: e.id, from: e.from, to: e.to })),
      normalizers: pipeline.normalizers.map((n) => ({ id: n.id, name: n.name, params: n.params, position: n.position })),
      models: pipeline.models.map((m) => ({ id: m.id, name: m.name, params: m.params || {}, position: m.position })),
      postprocessors: pipeline.postprocessors.map((p) => ({ id: p.id, name: p.name, params: p.params, position: p.position })),
    },
  };
}

// The PUT sent as the page is left. keepalive lets it finish after the page
// is gone, as a beacon would; a beacon can only POST.
function putPipelineOnUnload() {
  return fetch("/api/pipeline", {
    method: "PUT",
    keepalive: true,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildPipelineBody()),
  });
}

let applyTimer = null;

// Apply the pipeline 2 s after the last call.
export function scheduleApply() {
  if (applyTimer) clearTimeout(applyTimer);
  applyTimer = setTimeout(() => applyPipeline(), 2000);
}

// A refusal (a step its op's class will not take) or no answer is shown,
// with the server's reason: the page stays unapplied, and the unload
// beacon tries again.
async function applyPipeline() {
  const sending = changes;
  try {
    await postJSON("/api/pipeline", buildPipelineBody(), { method: "PUT" });
    appliedChanges = Math.max(appliedChanges, sending);
  } catch (err) {
    showMessage("Pipeline not applied: " + err.message, "error");
  }
}

// On leaving the page (the back button, a link) with changes the server has
// not had -- an apply still pending, or one that failed, or an edit that
// does not apply on its own -- send the pipeline in a request that outlives
// the page, instead of any apply still pending. A page left unchanged sends
// nothing: its pipeline is what the server gave it (moved by the layout at
// most), and sending it back would undo whatever changed the server's
// pipeline since it was loaded, such as Submit All on the dashboard.
export function syncOnUnload() {
  window.addEventListener("beforeunload", () => {
    if (applyTimer) clearTimeout(applyTimer);
    if (changes === appliedChanges) return;
    // Nothing is left to report a failure to.
    putPipelineOnUnload().catch(() => {});
  });
}
