// The canvas: the nodes where they sit, the edges between them drawn as SVG
// curves, and what the mouse does there -- dropping a palette entry adds a
// node, dragging a node's header moves it, dragging from a node's output port
// to another's input port draws an edge, clicking an edge deletes it.
//
// A node's element is built by nodes.js; the canvas places it and wires its
// header and ports.
import { showMessage } from "./messages.js";
import { createNodeElement } from "./nodes.js";
import {
  addEdge, addNodeTo, blockwiseSettings, changed, connectNewNode, defaultPosition, edited, findNode, nodeDefaults,
  NODE_TYPES, nodesOf, pipeline, removeNodeFrom, scheduleApply,
} from "./state.js";

const SVG_NS = "http://www.w3.org/2000/svg";

let draggedNode = null;     // the node whose header is being dragged
let selectedNode = null;    // the node element last brought to the front
let connectionDrag = null;  // an edge being drawn from an output port

const canvasContent = () => document.getElementById("canvas-content");

// Draw every node and edge afresh.
export function renderCanvas() {
  const canvas = canvasContent();
  canvas.innerHTML = "";

  const allNodes = NODE_TYPES.flatMap((type) => nodesOf(type).map((n) => ({ ...n, type })));
  if (allNodes.length === 0) {
    canvas.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🎨</div>
        <div class="empty-state-text">Drag INPUT/OUTPUT or other items from the sidebar to start building your pipeline</div>
      </div>`;
    return;
  }

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.classList.add("connections-svg");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");
  svg.style.minHeight = "600px";
  canvas.appendChild(svg);

  allNodes.forEach((node) => placeNode(node));
  renderConnections();
}

// Put a node's element on the canvas, with its header draggable and its
// ports live. `node` is a copy of the pipeline's node with its type added.
function placeNode(node) {
  const nodeEl = createNodeElement(node);
  nodeEl.querySelector(".node-input")?.addEventListener("mousedown", (event) => startConnection(event, node.id, "input"));
  nodeEl.querySelector(".node-output")?.addEventListener("mousedown", (event) => startConnection(event, node.id, "output"));
  canvasContent().appendChild(nodeEl);
  setupNodeDragging(nodeEl, node);
}

// A node of this type and name from the palette, at the drop position (or
// where its type goes), connected to its neighbours if its type is.
export async function addNode(type, name, dropPosition) {
  const id = `${type}-${Date.now()}`;
  const { x, y } = dropPosition || defaultPosition(type, (nodesOf(type) || []).length);

  let defaultParams = {};
  let modelConfig = null;
  if (type === "blockwise-config") {
    // The dashboard's current blockwise settings.
    try {
      const response = await fetch("/api/blockwise-config");
      defaultParams = blockwiseSettings(await response.json());
    } catch (error) {
      console.error("Error fetching blockwise config:", error);
      defaultParams = {};
    }
  } else {
    const defaults = nodeDefaults(type, name);
    if (defaults) {
      defaultParams = defaults.params;
      modelConfig = defaults.config || null;
    }
  }

  const node = { id, name, params: defaultParams, position: { x, y } };
  if (type === "model" && modelConfig) {
    node.config = modelConfig;
  }
  addNodeTo(type, node);
  connectNewNode(node, type);

  // Add just this node's element; the others stay as they are.
  placeNode({ ...node, type });
  renderConnections();
  edited();
  showMessage(`Added ${name}`, "success");
}

export function removeNode(id, type) {
  removeNodeFrom(type, id);
  const nodeEl = document.getElementById(`node-${id}`);
  if (nodeEl) nodeEl.remove();
  renderConnections();
  edited();
  showMessage("Node removed", "success");
}

function setupNodeDragging(nodeEl, node) {
  const header = nodeEl.querySelector(".node-header");
  header.addEventListener("mousedown", (e) => {
    // The pipeline's node, not the copy the element was drawn from.
    const actualNode = findNode(node.type, node.id);
    if (!actualNode) return;
    bringNodeToFront(nodeEl);
    draggedNode = {
      element: nodeEl,
      node: actualNode,
      startX: e.clientX,
      startY: e.clientY,
      originalX: actualNode.position?.x || 0,
      originalY: actualNode.position?.y || 0,
    };
    nodeEl.style.zIndex = 1000;
    nodeEl.style.cursor = "grabbing";
    header.style.cursor = "grabbing";
  });
  header.style.cursor = "grab";
}

// The node clicked last stays in front of the others (z-index 999; a node
// being dragged is at 1000).
function bringNodeToFront(nodeEl) {
  if (selectedNode && selectedNode !== nodeEl) {
    selectedNode.style.zIndex = "auto";
  }
  selectedNode = nodeEl;
  nodeEl.style.zIndex = 999;
}

// ---- edges ---------------------------------------------------------------

// Edges are drawn from an output port only.
function startConnection(event, nodeId, portType) {
  event.stopPropagation();
  event.preventDefault();
  if (portType !== "output") return;

  const dot = event.target;
  dot.classList.add("active");
  connectionDrag = { fromNode: nodeId, fromPort: portType, startDot: dot };

  const tempPath = document.createElementNS(SVG_NS, "path");
  tempPath.classList.add("connection-path", "dragging-connection");
  tempPath.id = "temp-connection";
  document.querySelector(".connections-svg").appendChild(tempPath);

  showMessage("Drag to an input to create connection", "info");
}

function updateConnectionDrag(event) {
  const tempPath = document.getElementById("temp-connection");
  if (!tempPath) return;
  const fromDot = document.getElementById(`node-${connectionDrag.fromNode}`).querySelector(".node-output");
  const fromRect = fromDot.getBoundingClientRect();
  const canvasRect = canvasContent().getBoundingClientRect();
  const x1 = fromRect.left + fromRect.width / 2 - canvasRect.left;
  const y1 = fromRect.top + fromRect.height / 2 - canvasRect.top;
  const x2 = event.clientX - canvasRect.left;
  const y2 = event.clientY - canvasRect.top;
  tempPath.setAttribute("d", createBezierPath(x1, y1, x2, y2));
}

function endConnection(event) {
  if (connectionDrag.startDot) {
    connectionDrag.startDot.classList.remove("active");
  }
  const tempPath = document.getElementById("temp-connection");
  if (tempPath) tempPath.remove();

  // Dropped on another node's input port?
  const target = document.elementFromPoint(event.clientX, event.clientY);
  if (target && target.classList.contains("node-input")) {
    const toNodeId = target.closest(".node-box").dataset.nodeid;
    if (toNodeId !== connectionDrag.fromNode) {
      if (addEdge(connectionDrag.fromNode, toNodeId, `edge-${Date.now()}`)) {
        renderConnections();
        edited();
        showMessage("Connection created", "success");
      } else {
        showMessage("Connection already exists", "info");
      }
    }
  }
  connectionDrag = null;
}

function createBezierPath(x1, y1, x2, y2) {
  const curveStrength = Math.min(Math.abs(x2 - x1) * 0.5, 150);
  const cx1 = x1 + curveStrength;
  const cy1 = y1;
  const cx2 = x2 - curveStrength;
  const cy2 = y2;
  return `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`;
}

// A port's centre in canvas-content's coordinates. The port and
// canvas-content scroll together, so the difference of their client rects
// does not depend on the scroll.
function getDotPosition(dot) {
  const containerRect = canvasContent().getBoundingClientRect();
  const dotRect = dot.getBoundingClientRect();
  return {
    x: dotRect.left + dotRect.width / 2 - containerRect.left,
    y: dotRect.top + dotRect.height / 2 - containerRect.top,
  };
}

// Redraw every edge (not one being drawn), and grow the canvas to hold every node.
export function renderConnections() {
  const svg = document.querySelector(".connections-svg");
  if (!svg) return;
  svg.querySelectorAll(".connection-path:not(.dragging-connection)").forEach((p) => p.remove());

  const canvas = canvasContent();
  let maxRight = 600, maxBottom = 600;
  canvas.querySelectorAll(".node-box").forEach((box) => {
    const right = box.offsetLeft + box.offsetWidth;
    const bottom = box.offsetTop + box.offsetHeight;
    if (right > maxRight) maxRight = right;
    if (bottom > maxBottom) maxBottom = bottom;
  });
  // The SVG is 100% of canvas-content, so this makes it cover every node too.
  canvas.style.minWidth = (maxRight + 100) + "px";
  canvas.style.minHeight = (maxBottom + 100) + "px";

  pipeline.edges.forEach((edge) => {
    const fromNodeEl = document.getElementById(`node-${edge.from}`);
    const toNodeEl = document.getElementById(`node-${edge.to}`);
    if (!fromNodeEl || !toNodeEl) return;
    const fromDot = fromNodeEl.querySelector(".node-output");
    const toDot = toNodeEl.querySelector(".node-input");
    if (!fromDot || !toDot) return;

    const from = getDotPosition(fromDot);
    const to = getDotPosition(toDot);
    const path = document.createElementNS(SVG_NS, "path");
    path.classList.add("connection-path");
    path.setAttribute("d", createBezierPath(from.x, from.y, to.x, to.y));
    path.dataset.edgeId = edge.id;
    path.style.pointerEvents = "stroke";
    path.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteConnection(edge.id);
    });
    svg.appendChild(path);
  });
}

function deleteConnection(edgeId) {
  pipeline.edges = pipeline.edges.filter((e) => e.id !== edgeId);
  renderConnections();
  edited();
  showMessage("Connection deleted", "success");
}

// ---- layout --------------------------------------------------------------

// Arrange the nodes left to right by their place in the graph (longest-path
// layering over the edges), each column as wide as its widest node and
// centred on the tallest, with the blockwise nodes in a row below. It
// measures the nodes' elements, so the canvas must be drawn first.
export function autoLayoutNodes() {
  const COL_GAP = 80;
  const ROW_SPACING = 40;
  const LEFT_MARGIN = 40;
  const TOP_MARGIN = 40;

  const allNodes = [
    ...pipeline.inputs,
    ...pipeline.normalizers,
    ...pipeline.models,
    ...pipeline.postprocessors,
    ...pipeline.outputs,
  ];
  if (allNodes.length === 0) return;

  const nodeById = {};
  allNodes.forEach((n) => { nodeById[n.id] = n; });

  const nodeSize = {};
  allNodes.forEach((n) => {
    const el = document.getElementById(`node-${n.id}`);
    nodeSize[n.id] = el ? { w: el.offsetWidth, h: el.offsetHeight } : { w: 300, h: 120 };
  });

  const incomingCount = {};
  const outgoing = {};
  allNodes.forEach((n) => {
    incomingCount[n.id] = 0;
    outgoing[n.id] = [];
  });
  pipeline.edges.forEach((e) => {
    if (nodeById[e.from] && nodeById[e.to]) {
      outgoing[e.from].push(e.to);
      incomingCount[e.to] = (incomingCount[e.to] || 0) + 1;
    }
  });

  // Layers by topological BFS, each node one past its deepest parent.
  const layer = {};
  allNodes.forEach((n) => { layer[n.id] = 0; });
  const queue = allNodes.filter((n) => incomingCount[n.id] === 0).map((n) => n.id);
  const visited = new Set();
  while (queue.length > 0) {
    const nid = queue.shift();
    if (visited.has(nid)) continue;
    visited.add(nid);
    outgoing[nid].forEach((childId) => {
      layer[childId] = Math.max(layer[childId], layer[nid] + 1);
      incomingCount[childId]--;
      if (incomingCount[childId] <= 0) {
        queue.push(childId);
      }
    });
  }
  // Nodes the walk never reached (in a cycle, or after one) go in the first column.
  allNodes.forEach((n) => {
    if (!visited.has(n.id)) {
      layer[n.id] = 0;
      visited.add(n.id);
    }
  });

  const layers = {};
  allNodes.forEach((n) => {
    const l = layer[n.id];
    if (!layers[l]) layers[l] = [];
    layers[l].push(n);
  });
  const sortedLayerKeys = Object.keys(layers).map(Number).sort((a, b) => a - b);

  const colHeights = {};
  sortedLayerKeys.forEach((k) => {
    const nodes = layers[k];
    colHeights[k] = nodes.reduce((sum, n) => sum + nodeSize[n.id].h, 0) + (nodes.length - 1) * ROW_SPACING;
  });
  const maxColHeight = Math.max(...Object.values(colHeights));

  const colX = {};
  let currentX = LEFT_MARGIN;
  sortedLayerKeys.forEach((k) => {
    colX[k] = currentX;
    currentX += Math.max(...layers[k].map((n) => nodeSize[n.id].w)) + COL_GAP;
  });

  const moveTo = (node, position) => {
    node.position = position;
    const el = document.getElementById(`node-${node.id}`);
    if (el) {
      el.style.left = position.x + "px";
      el.style.top = position.y + "px";
    }
  };
  sortedLayerKeys.forEach((k) => {
    let y = TOP_MARGIN + (maxColHeight - colHeights[k]) / 2;
    layers[k].forEach((node) => {
      moveTo(node, { x: colX[k], y });
      y += nodeSize[node.id].h + ROW_SPACING;
    });
  });

  if (pipeline.blockwise_config && pipeline.blockwise_config.length > 0) {
    const bottomY = TOP_MARGIN + maxColHeight + 60;
    pipeline.blockwise_config.forEach((node, i) => {
      const size = nodeSize[node.id] || { w: 350 };
      moveTo(node, { x: LEFT_MARGIN + i * (size.w + COL_GAP), y: bottomY });
    });
  }

  renderConnections();
}

// ---- the mouse -----------------------------------------------------------

export function initCanvas() {
  const canvas = document.querySelector(".canvas");
  canvas.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  canvas.addEventListener("drop", (e) => {
    e.preventDefault();
    const rect = canvasContent().getBoundingClientRect();
    addNode(e.dataTransfer.getData("type"), e.dataTransfer.getData("name"), {
      x: Math.max(0, e.clientX - rect.left),
      y: Math.max(0, e.clientY - rect.top),
    });
  });

  document.addEventListener("mousemove", (e) => {
    if (draggedNode) {
      const dx = e.clientX - draggedNode.startX;
      const dy = e.clientY - draggedNode.startY;
      draggedNode.element.style.left = Math.max(0, draggedNode.originalX + dx) + "px";
      draggedNode.element.style.top = Math.max(0, draggedNode.originalY + dy) + "px";
      renderConnections();
    }
    if (connectionDrag) {
      updateConnectionDrag(e);
    }
  });

  document.addEventListener("mouseup", (e) => {
    if (draggedNode) {
      const dx = e.clientX - draggedNode.startX;
      const dy = e.clientY - draggedNode.startY;
      draggedNode.node.position = { x: Math.max(0, draggedNode.originalX + dx), y: Math.max(0, draggedNode.originalY + dy) };
      changed();
      scheduleApply();
      // It stays in front (999), as the node selected last.
      draggedNode.element.style.zIndex = 999;
      draggedNode.element.style.cursor = "auto";
      draggedNode = null;
      renderConnections();
    }
    if (connectionDrag) {
      endConnection(e);
    }
  });
}
