// Export YAML and Import YAML: the pipeline as a file, and back.
//
// The file has a section per node list, and the edges. Each node has its own
// fields first (id, name), then what it holds, then its position:
//   inputs, outputs: id, then the node's params as its own fields
//     (dataset_path, bounding_boxes, separate_bounding_boxes_zarrs,
//     output_channels, ...);
//   input_normalizers, postprocessors, blockwise_config: id, name, params;
//   models: id, name, then its config's fields (type, checkpoint_path,
//     channels, ...), or its params when it has no config;
//   edges: id, from, to.
// An import may also be the builder's own structure as JSON (inputs with
// params, models with a config, normalizers by name). Importing replaces the
// pipeline; its edges are rebuilt from the node order, and its nodes laid
// out afresh.
import { postJSON } from "../lib/api.js";
import { CORE_SCHEMA, dump, load } from "../vendor/js-yaml.js";
import { autoLayoutNodes, renderCanvas } from "./canvas.js";
import { showMessage } from "./messages.js";
import { autoConnectNodes, blockwiseSettings, datasetPath, defaultPosition, edited, pipeline, replacePipeline } from "./state.js";

// Blocks down to a node's own fields, and anything nested deeper (a box's
// offset and shape, a parameter's list) inline; no line folding, no
// anchors, and whatever is not plain data (undefined) left out.
const DUMP_OPTIONS = { flowLevel: 4, lineWidth: -1, noRefs: true, skipInvalid: true, quotingType: '"' };

// A node's fields but these.
const fieldsOf = (node, ...skip) => Object.fromEntries(Object.entries(node).filter(([key]) => !skip.includes(key)));

// The pipeline in the file's layout, the sections that have nodes only. The
// first INPUT gets the dashboard's dataset when it has no path of its own.
function toFileLayout() {
  const ioNode = (n, params) => ({ id: n.id, ...params, position: n.position });
  const opNode = (n) => ({
    id: n.id, name: n.name, ...(Object.keys(n.params || {}).length > 0 ? { params: n.params } : {}), position: n.position,
  });
  const sections = {
    inputs: pipeline.inputs.map((n, index) => ioNode(n, index === 0 && !n.params?.dataset_path && datasetPath
      ? { ...n.params, dataset_path: datasetPath } : n.params)),
    outputs: pipeline.outputs.map((n) => ioNode(n, n.params)),
    input_normalizers: pipeline.normalizers.map(opNode),
    models: pipeline.models.map((m) => (m.config && typeof m.config === "object"
      ? { id: m.id, name: m.name, ...fieldsOf(m.config, "name"), position: m.position }
      : opNode(m))),
    postprocessors: pipeline.postprocessors.map(opNode),
    blockwise_config: pipeline.blockwise_config.map(opNode),
    edges: pipeline.edges.map((e) => ({ id: e.id, from: e.from, to: e.to })),
  };
  return Object.fromEntries(Object.entries(sections).filter(([, nodes]) => nodes.length > 0));
}

// Each section dumped on its own, so a blank line parts them.
function exportYAML() {
  const text = Object.entries(toFileLayout()).map(([section, nodes]) => dump({ [section]: nodes }, DUMP_OPTIONS)).join("\n");
  downloadFile(text, "pipeline.yaml", "text/yaml");
}

function downloadFile(content, filename, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
  showMessage("Exported as " + filename, "success");
}

function importFile() {
  const fileInput = document.getElementById("import-file");
  const file = fileInput.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const content = e.target.result;
      // YAML's core schema is plain data: a date-like string stays a string.
      const data = /\.ya?ml$/.test(file.name) ? fromFileLayout(load(content, { schema: CORE_SCHEMA })) : JSON.parse(content);
      replacePipeline(pipelineFromFile(data));
      autoConnectNodes();
      renderCanvas();
      autoLayoutNodes();

      // An imported blockwise config becomes the dashboard's.
      if (pipeline.blockwise_config.length > 0) {
        const blockwiseParams = pipeline.blockwise_config[0].params;
        if (blockwiseParams) {
          // A refusal (a count that is not a whole number) is an error with
          // the server's reason, not "updated".
          postJSON("/api/blockwise-config", blockwiseSettings(blockwiseParams)).then(() => {
            showMessage("✓ Blockwise config updated from import", "success");
          }).catch((err) => {
            showMessage("Blockwise config not updated from import: " + err.message, "error");
          });
        }
      }

      edited({ apply: false });
      showMessage("Pipeline imported successfully", "success");
      fileInput.value = "";
    } catch (err) {
      showMessage("Import failed: " + err.message, "error");
    }
  };
  reader.readAsText(file);
}

// The pipeline's lists from an imported file's, every node with an id and a
// position. An op may be given by its name alone; a model given with its
// config's fields as its own (type, ...) gets them as its config and params.
function pipelineFromFile(data) {
  const op = (prefix, type) => (n, i) => ({
    id: n.id || `${prefix}-${Date.now()}-${i}`,
    name: typeof n === "string" ? n : (n.name || n),
    params: n.params || {},
    position: n.position || defaultPosition(type, i),
  });
  return {
    inputs: (data.inputs || []).map((n, i) => ({
      id: n.id || `input-${Date.now()}-${i}`,
      name: "INPUT",
      params: n.params || { dataset_path: datasetPath },
      position: n.position || defaultPosition("input", i),
    })),
    outputs: (data.outputs || []).map((n, i) => ({
      id: n.id || `output-${Date.now()}-${i}`,
      name: "OUTPUT",
      params: n.params || {},
      position: n.position || defaultPosition("output", i),
    })),
    normalizers: (data.input_normalizers || data.normalizers || []).map(op("norm", "normalizer")),
    models: (data.models || []).map((m, i) => {
      const model = {
        id: m.id || `model-${Date.now()}-${i}`,
        name: typeof m === "string" ? m : (m.name || m),
        params: m.params || m.config || {},
        position: m.position || defaultPosition("model", i),
      };
      if (m.config && typeof m.config === "object") {
        model.config = m.config;
      } else if (m.type) {
        model.config = { ...m };
        delete model.config.id;
        delete model.config.params;
        delete model.config.position;
        model.params = { ...model.config };
      }
      return model;
    }),
    postprocessors: (data.postprocessors || []).map(op("post", "postprocessor")),
    blockwise_config: (data.blockwise_config || []).map((c, i) => ({
      id: c.id || `blockwise-${Date.now()}-${i}`,
      name: "Blockwise Configuration",
      params: c.params,
      position: c.position || defaultPosition("blockwise-config", i),
    })),
    edges: data.edges || [],
  };
}

// A YAML file's nodes in the builder's structure: an INPUT's or OUTPUT's own
// fields are its params, and a model's are its config and, for display, its
// params, with its channels as a list. A node written with params: keeps
// them, and one written as a bare name stays one. A file that is not a
// mapping (an empty one, say) is an empty pipeline.
function fromFileLayout(doc) {
  if (!doc || typeof doc !== "object") return {};
  const isNode = (n) => n && typeof n === "object";
  const withParams = (n) => {
    if (!isNode(n) || n.params) return n;
    const params = fieldsOf(n, "id", "name", "position");
    return Object.keys(params).length > 0 ? { ...n, params } : n;
  };
  const model = (m) => {
    if (!isNode(m)) return m;
    const config = fieldsOf(m, "id", "name", "position", "params");
    if (Object.keys(config).length === 0) return m;
    if ("channels" in config && !Array.isArray(config.channels)) config.channels = [config.channels];
    return { id: m.id, name: m.name, position: m.position, config, params: { ...config } };
  };
  return {
    ...doc,
    inputs: (doc.inputs || []).map(withParams),
    outputs: (doc.outputs || []).map(withParams),
    models: (doc.models || []).map(model),
  };
}

export function initIo() {
  document.getElementById("export-yaml-btn").addEventListener("click", exportYAML);
  document.getElementById("import-file").addEventListener("change", importFile);
}
