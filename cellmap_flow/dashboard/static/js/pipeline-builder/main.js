// The pipeline builder page (templates/pipeline_builder_v2.html): a node
// editor for INPUT -> normalizers -> models -> postprocessors -> OUTPUT, and
// a blockwise job's settings, applied to the dashboard as it is edited.
//
//   state.js      the pipeline, the palette's defaults, applying to the server
//   canvas.js     drawing nodes and edges; drops, drags, layout
//   nodes.js      a node's element and its controls
//   palette.js    the sidebar
//   io.js         Export YAML / Import YAML
//   bbx.js        an INPUT node's bounding boxes
//   output-channels.js, model-config-modal.js   the two other dialogs
//   blockwise.js  the Blockwise bar
//   log-panel.js  the Flow Logs panel
//   dialogs.js, messages.js   the dialogs' backdrop, and toasts
//
// The modules use each other's functions freely (the canvas draws nodes
// whose buttons redraw the canvas), which is safe because none of them runs
// anything when it is imported: this module starts the page. A module runs
// once the page is parsed and before DOMContentLoaded, so every element is
// there.
import { initBbx } from "./bbx.js";
import { initBlockwise } from "./blockwise.js";
import { autoLayoutNodes, initCanvas, renderCanvas } from "./canvas.js";
import { initDialogs } from "./dialogs.js";
import { initIo } from "./io.js";
import { initLogPanel } from "./log-panel.js";
import { initModelConfigModal } from "./model-config-modal.js";
import { initOutputChannels } from "./output-channels.js";
import { initPalette } from "./palette.js";
import { loadPipeline, syncOnUnload } from "./state.js";

loadPipeline();
syncOnUnload();
initLogPanel();
initCanvas();
initPalette();
renderCanvas();
autoLayoutNodes();  // measures the nodes, so after they are drawn
initBlockwise();
initDialogs();
initOutputChannels();
initBbx();
initModelConfigModal();
initIo();
document.getElementById("auto-arrange-btn").addEventListener("click", autoLayoutNodes);
