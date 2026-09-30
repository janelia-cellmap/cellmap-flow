// Export YAML and Import YAML: the pipeline as a file, and back.
//
// The file has a section per node list (inputs, outputs, input_normalizers,
// models, postprocessors, blockwise_config) and the edges. An import may
// also be the same structure as JSON. Importing replaces the pipeline; its
// edges are rebuilt from the node order, and its nodes laid out afresh.
import { autoLayoutNodes, renderCanvas } from "./canvas.js";
import { showMessage } from "./messages.js";
import { autoConnectNodes, blockwiseSettings, datasetPath, defaultPosition, edited, pipeline, replacePipeline } from "./state.js";

function exportYAML() {
  let yaml = "";

  // Inputs with dataset_path and bounding boxes
  if (pipeline.inputs.length > 0) {
    yaml += "inputs:\n";
    pipeline.inputs.forEach((n, index) => {
      yaml += `  - id: ${n.id}\n`;
      // Add dataset_path to the first input node
      if (index === 0) {
        const path = n.params?.dataset_path || datasetPath;
        if (path) {
          yaml += `    dataset_path: ${path}\n`;
        }
      }
      if (n.params?.bounding_boxes && n.params.bounding_boxes.length > 0) {
        yaml += "    bounding_boxes:\n";
        n.params.bounding_boxes.forEach((bbox) => {
          yaml += `      - offset: [${bbox.offset.join(", ")}]\n`;
          yaml += `        shape: [${bbox.shape.join(", ")}]\n`;
        });
      }
      yaml += "    position:\n";
      yaml += `      x: ${n.position.x}\n`;
      yaml += `      y: ${n.position.y}\n`;
    });
    yaml += "\n";
  }

  if (pipeline.outputs.length > 0) {
    yaml += "outputs:\n";
    pipeline.outputs.forEach((n) => {
      yaml += `  - id: ${n.id}\n`;
      if (n.params && Object.keys(n.params).length > 0) {
        Object.entries(n.params).forEach(([k, v]) => {
          yaml += `    ${k}: ${JSON.stringify(v)}\n`;
        });
      }
      yaml += "    position:\n";
      yaml += `      x: ${n.position.x}\n`;
      yaml += `      y: ${n.position.y}\n`;
    });
    yaml += "\n";
  }

  const exportOps = (section, nodes) => {
    if (nodes.length === 0) return;
    yaml += `${section}:\n`;
    nodes.forEach((n) => {
      yaml += `  - id: ${n.id}\n`;
      yaml += `    name: ${n.name}\n`;
      if (Object.keys(n.params).length > 0) {
        yaml += "    params:\n";
        Object.entries(n.params).forEach(([k, v]) => {
          yaml += `      ${k}: ${JSON.stringify(v)}\n`;
        });
      }
      yaml += "    position:\n";
      yaml += `      x: ${n.position.x}\n`;
      yaml += `      y: ${n.position.y}\n`;
    });
    yaml += "\n";
  };

  exportOps("input_normalizers", pipeline.normalizers);

  if (pipeline.models.length > 0) {
    yaml += "models:\n";
    pipeline.models.forEach((m) => {
      yaml += `  - id: ${m.id}\n`;
      yaml += `    name: ${m.name}\n`;
      // A full config (the server's ModelConfig.to_dict()) goes in as the
      // node's own fields; without one, its params.
      if (m.config && typeof m.config === "object") {
        Object.entries(m.config).forEach(([k, v]) => {
          if (k !== "name") {
            if (Array.isArray(v)) {
              yaml += `    ${k}: [${v.join(", ")}]\n`;
            } else {
              yaml += `    ${k}: ${JSON.stringify(v)}\n`;
            }
          }
        });
      } else if (Object.keys(m.params || {}).length > 0) {
        yaml += "    params:\n";
        Object.entries(m.params).forEach(([k, v]) => {
          yaml += `      ${k}: ${JSON.stringify(v)}\n`;
        });
      }
      yaml += "    position:\n";
      yaml += `      x: ${m.position.x}\n`;
      yaml += `      y: ${m.position.y}\n`;
    });
    yaml += "\n";
  }

  exportOps("postprocessors", pipeline.postprocessors);
  exportOps("blockwise_config", pipeline.blockwise_config);

  if (pipeline.edges.length > 0) {
    yaml += "edges:\n";
    pipeline.edges.forEach((e) => {
      yaml += `  - id: ${e.id}\n`;
      yaml += `    from: ${e.from}\n`;
      yaml += `    to: ${e.to}\n`;
    });
  }

  downloadFile(yaml, "pipeline.yaml", "text/yaml");
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
      const data = file.name.endsWith(".yaml") ? parseYAML(content) : JSON.parse(content);
      replacePipeline(pipelineFromFile(data));
      autoConnectNodes();
      renderCanvas();
      autoLayoutNodes();

      // An imported blockwise config becomes the dashboard's.
      if (pipeline.blockwise_config.length > 0) {
        const blockwiseParams = pipeline.blockwise_config[0].params;
        if (blockwiseParams) {
          fetch("/api/blockwise-config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(blockwiseSettings(blockwiseParams)),
          }).then((r) => r.json()).then((data) => {
            console.log("Blockwise config synced from import:", data);
            showMessage("✓ Blockwise config updated from import", "success");
          }).catch((err) => {
            console.error("Error syncing blockwise config from import:", err);
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

// A reader for the layout exportYAML writes, and only that.
function parseYAML(yaml) {
  const result = {
    inputs: [],
    outputs: [],
    input_normalizers: [],
    models: [],
    postprocessors: [],
    blockwise_config: [],
    edges: [],
  };
  let currentSection = null;
  let currentItem = null;
  let inParams = false;
  let inPosition = false;
  let inBoundingBoxes = false;
  let currentBBox = null;
  const SECTIONS = ["inputs", "outputs", "input_normalizers", "models", "postprocessors", "blockwise_config", "edges"];
  // "[a, b, 1]" -> ["a", "b", 1]
  const arrayItems = (content) => content.split(",").map((item) => {
    const trimmed = item.trim();
    const num = parseFloat(trimmed);
    return isNaN(num) ? trimmed : num;
  });
  const intItems = (content) => content.split(",").map((s) => {
    const num = parseFloat(s.trim());
    return isNaN(num) ? 0 : parseInt(num);
  });

  yaml.split("\n").forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) return;

    if (SECTIONS.some((section) => trimmed === `${section}:`)) {
      currentSection = trimmed.slice(0, -1);
      inParams = false;
      inPosition = false;
      inBoundingBoxes = false;
    } else if (trimmed === "params:") {
      inParams = true;
      inPosition = false;
      inBoundingBoxes = false;
      if (currentItem && !currentItem.params) {
        currentItem.params = {};
      }
    } else if (trimmed === "position:") {
      inPosition = true;
      inParams = false;
      inBoundingBoxes = false;
      if (currentItem && !currentItem.position) {
        currentItem.position = {};
      }
    } else if (trimmed === "bounding_boxes:") {
      inBoundingBoxes = true;
      inParams = false;
      inPosition = false;
      if (currentItem && !currentItem.params) {
        currentItem.params = {};
      }
      if (currentItem && !currentItem.params.bounding_boxes) {
        currentItem.params.bounding_boxes = [];
      }
    } else if (trimmed.startsWith("- ")) {
      if (inBoundingBoxes && trimmed.startsWith("- offset:")) {
        const offsetMatch = trimmed.match(/- offset:\s*\[(.*?)\]/);
        if (offsetMatch) {
          currentBBox = { offset: intItems(offsetMatch[1]), shape: [] };
          currentItem.params.bounding_boxes.push(currentBBox);
        }
      } else if (inBoundingBoxes) {
        // A new item after the boxes.
        const match = trimmed.match(/^- (\w+):\s*(.*)$/);
        if (match) {
          const [, key, value] = match;
          currentItem = { [key]: value.trim() };
          if (currentSection) result[currentSection].push(currentItem);
          inBoundingBoxes = false;
          inParams = false;
          inPosition = false;
        }
      } else {
        const match = trimmed.match(/^- (\w+):\s*(.*)$/);
        currentItem = match ? { [match[1]]: match[2].trim() } : { name: trimmed.replace("- ", "").trim() };
        if (currentSection) result[currentSection].push(currentItem);
        inParams = false;
        inPosition = false;
        inBoundingBoxes = false;
      }
    } else if (trimmed.includes(":")) {
      const colonIdx = trimmed.indexOf(":");
      const key = trimmed.slice(0, colonIdx).trim();
      const value = trimmed.slice(colonIdx + 1).trim();
      if (!currentItem) return;

      if (inBoundingBoxes && key === "shape" && currentBBox) {
        const shapeMatch = value.match(/\[(.*?)\]/);
        if (shapeMatch) {
          currentBBox.shape = intItems(shapeMatch[1]);
        }
      } else if (inPosition) {
        currentItem.position[key] = parseFloat(value) || 0;
      } else if (inParams) {
        currentItem.params = currentItem.params || {};
        try {
          currentItem.params[key] = JSON.parse(value);
        } catch {
          const arrayMatch = value.match(/^\[(.*)\]$/);
          currentItem.params[key] = arrayMatch ? arrayItems(arrayMatch[1]) : value;
        }
      } else {
        try {
          currentItem[key] = JSON.parse(value);
        } catch {
          const arrayMatch = value.match(/^\[(.*)\]$/);
          currentItem[key] = arrayMatch ? arrayItems(arrayMatch[1]) : value;
        }

        // An INPUT or OUTPUT node's dataset_path is its param.
        if ((currentSection === "inputs" || currentSection === "outputs") && key === "dataset_path") {
          if (!currentItem.params) currentItem.params = {};
          currentItem.params.dataset_path = currentItem[key];
        }

        // A model's fields are its config, and its params for display.
        if (currentSection === "models" && key !== "id" && key !== "position" && key !== "params" && key !== "name") {
          currentItem.config = currentItem.config || {};
          try {
            let parsedValue = JSON.parse(value);
            if (key === "channels" && !Array.isArray(parsedValue)) {
              parsedValue = [parsedValue];
            }
            currentItem.config[key] = parsedValue;
          } catch {
            if (["channels", "input_size", "output_size", "input_voxel_size", "output_voxel_size"].includes(key)) {
              const arrayMatch = value.match(/^\[(.*)\]$/);
              currentItem.config[key] = arrayMatch ? arrayItems(arrayMatch[1]) : [value];
            } else {
              currentItem.config[key] = value;
            }
          }
          currentItem.params = currentItem.params || {};
          currentItem.params[key] = currentItem.config[key];
        }
      }
    }
  });

  return result;
}

export function initIo() {
  document.getElementById("export-yaml-btn").addEventListener("click", exportYAML);
  document.getElementById("import-file").addEventListener("change", importFile);
}
