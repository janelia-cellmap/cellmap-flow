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
// server (POST /api/pipeline/apply, 2 s after the last edit) and the blockwise
// steps, which ran on the pipeline as it was, start over.
import { pageData } from "../lib/page-data.js";

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
  return pipeline[LISTS[type]];
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
  if (LISTS[type]) pipeline[LISTS[type]] = nodesOf(type).filter((n) => n.id !== id);
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

// The page's starting pipeline: what the builder last applied, or the live
// chain (pageData().pipeline). Every node gets an id and a position, and
// there is always an INPUT node, with the dashboard's dataset if it has no
// path of its own. The edges are then rebuilt from the node order.
export function loadPipeline() {
  const start = PAGE.pipeline;
  const inputs = (start.inputs || []).map((n, i) => ({
    id: n.id || `input-${i}-${Date.now()}`,
    name: "INPUT",
    params: n.params || { dataset_path: datasetPath },
    position: n.position || defaultPosition("input", i),
  }));
  if (inputs.length === 0) {
    inputs.push({
      id: `input-0-${Date.now()}`,
      name: "INPUT",
      params: { dataset_path: datasetPath },
      position: defaultPosition("input", 0),
    });
  } else if (!inputs[0].params?.dataset_path) {
    inputs[0].params = inputs[0].params || {};
    inputs[0].params.dataset_path = datasetPath;
  }
  const simple = (prefix, type) => (n, i) => ({
    id: n.id || `${prefix}-${i}-${Date.now()}`,
    name: n.name,
    params: n.params || {},
    position: n.position || defaultPosition(type, i),
  });
  replacePipeline({
    inputs,
    outputs: (start.outputs || []).map((n, i) => ({
      id: n.id || `output-${i}-${Date.now()}`,
      name: "OUTPUT",
      params: n.params || {},
      position: n.position || defaultPosition("output", i),
    })),
    normalizers: (start.normalizers || []).map(simple("norm", "normalizer")),
    models: (start.models || []).map(loadModel),
    postprocessors: (start.postprocessors || []).map(simple("post", "postprocessor")),
    blockwise_config: [],
    edges: start.edges || [],
  });
  autoConnectNodes();
}

function loadModel(m, i) {
  const model = {
    id: m.id || `model-${i}-${Date.now()}`,
    name: m.name,
    params: m.params || {},
    position: m.position || defaultPosition("model", i),
  };
  // Keep the config (the server's ModelConfig.to_dict()), with its channel
  // names as an array: 'channels' (FlyModel etc.) or 'channels_names'
  // (HuggingFace), which may come as a JSON string. 'channels' is what the
  // rest of the builder reads.
  if (m.config && typeof m.config === "object") {
    model.config = m.config;
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
      Object.entries(m.config).forEach(([key, value]) => {
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

// The edges a pipeline in the usual order has, replacing any it had: INPUT ->
// the normalizers in a chain -> every model -> the postprocessors in a chain
// -> OUTPUT, skipping any stage that has no nodes.
export function autoConnectNodes() {
  pipeline.edges = [];
  const { inputs, normalizers, models, postprocessors, outputs } = pipeline;

  if (inputs.length > 0 && normalizers.length > 0) {
    addEdge(inputs[0].id, normalizers[0].id);
  }
  for (let i = 0; i < normalizers.length - 1; i++) {
    addEdge(normalizers[i].id, normalizers[i + 1].id);
  }
  const sourceForModels = normalizers.length > 0
    ? normalizers[normalizers.length - 1].id
    : (inputs.length > 0 ? inputs[0].id : null);
  if (sourceForModels) {
    models.forEach((model) => addEdge(sourceForModels, model.id));
  }
  if (postprocessors.length > 0) {
    models.forEach((model) => addEdge(model.id, postprocessors[0].id));
  } else if (outputs.length > 0) {
    models.forEach((model) => addEdge(model.id, outputs[0].id));
  }
  for (let i = 0; i < postprocessors.length - 1; i++) {
    addEdge(postprocessors[i].id, postprocessors[i + 1].id);
  }
  if (postprocessors.length > 0 && outputs.length > 0) {
    addEdge(postprocessors[postprocessors.length - 1].id, outputs[0].id);
  }
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

// listener() runs after every edit of the pipeline.
export function onEdit(listener) {
  editListeners.push(listener);
}

// The pipeline changed. Unless `apply` is false the change is applied to the
// server; the edits that pass false (bounding boxes, the separate-zarrs box,
// an import) have never been applied on their own, and reach the server with
// the next apply or the unload beacon.
export function edited({ apply = true } = {}) {
  if (apply) scheduleApply();
  editListeners.forEach((listener) => listener());
}

// The pipeline as /api/pipeline/apply takes it. It is also what the page
// starts from on its next load. Model configs are left out.
function buildApplyPayload() {
  return {
    input_normalizers: pipeline.normalizers.map((n) => ({ id: n.id, name: n.name, params: n.params, position: n.position })),
    postprocessors: pipeline.postprocessors.map((p) => ({ id: p.id, name: p.name, params: p.params, position: p.position })),
    models: pipeline.models.map((m) => ({ id: m.id, name: m.name, params: m.params || {}, position: m.position })),
    inputs: pipeline.inputs.map((i) => ({ id: i.id, params: i.params, position: i.position })),
    outputs: pipeline.outputs.map((o) => ({ id: o.id, params: o.params, position: o.position })),
    edges: pipeline.edges.map((e) => ({ id: e.id, from: e.from, to: e.to })),
  };
}

let applyTimer = null;

// Apply the pipeline 2 s after the last call.
export function scheduleApply() {
  if (applyTimer) clearTimeout(applyTimer);
  applyTimer = setTimeout(() => applyPipeline(), 2000);
}

async function applyPipeline() {
  const payload = buildApplyPayload();
  try {
    const response = await fetch("/api/pipeline/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      console.error("Pipeline sync error:", result.error || "Unknown error");
    }
  } catch (err) {
    console.error("Pipeline sync failed:", err.message);
  }
}

// On leaving the page (the back button, a link), apply the pipeline once more
// by beacon, which outlives the page, instead of any apply still pending.
export function syncOnUnload() {
  window.addEventListener("beforeunload", () => {
    if (applyTimer) clearTimeout(applyTimer);
    const payload = buildApplyPayload();
    navigator.sendBeacon("/api/pipeline/apply", new Blob([JSON.stringify(payload)], { type: "application/json" }));
  });
}
