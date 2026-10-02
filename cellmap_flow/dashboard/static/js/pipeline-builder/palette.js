// The sidebar: the nodes to drag onto the canvas, in sections that open and
// close. A drag carries the node's type and name (see canvas.js's drop).
import { h } from "../lib/dom.js";
import { palette } from "./state.js";

function handleDragStart(e) {
  e.dataTransfer.effectAllowed = "copy";
  e.dataTransfer.setData("type", e.target.dataset.type);
  e.dataTransfer.setData("name", e.target.dataset.name);
}

// A palette entry for a node of this type and name, at the end of its section.
export function addPaletteItem(sectionId, type, name) {
  document.getElementById(`${sectionId}-items`).appendChild(
    h("div", { class: "library-item", draggable: "true", dataset: { type, name }, onDragstart: handleDragStart }, name),
  );
}

function toggleSection(btn, sectionId) {
  const items = document.getElementById(`${sectionId}-items`);
  const toggle = document.getElementById(`${sectionId}-toggle`);
  items.classList.toggle("show");
  toggle.textContent = items.classList.contains("show") ? "▲" : "▼";
  btn.classList.toggle("active");
}

// Open a section (the Blockwise button does, to show where its node came from).
export function showSection(sectionId) {
  const items = document.getElementById(`${sectionId}-items`);
  if (items && !items.classList.contains("show")) {
    items.classList.add("show");
    const toggle = document.getElementById(`${sectionId}-toggle`);
    if (toggle) toggle.textContent = "▲";
  }
}

export function initPalette() {
  // INPUT, OUTPUT and the blockwise config are in the page's markup.
  document.querySelectorAll("#io-items .library-item").forEach((item) => item.addEventListener("dragstart", handleDragStart));
  const nameOf = (item) => (typeof item === "string" ? item : item.name);
  palette.normalizer.forEach((item) => addPaletteItem("normalizers", "normalizer", nameOf(item)));
  palette.model.forEach((item) => addPaletteItem("models", "model", typeof item === "string" ? item : item.name || item));
  palette.postprocessor.forEach((item) => addPaletteItem("postprocessors", "postprocessor", nameOf(item)));
  document.querySelectorAll("#config-items .library-item").forEach((item) => item.addEventListener("dragstart", handleDragStart));
  document.querySelectorAll(".section-toggle[data-section]").forEach((btn) => {
    btn.addEventListener("click", () => toggleSection(btn, btn.dataset.section));
  });
}
