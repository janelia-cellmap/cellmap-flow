// Building DOM from data without parsing it as HTML.

const ESC_MAP = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

// Text made safe to put inside HTML markup or a quoted attribute. For new
// code, h() is the better tool: it never builds markup at all.
export function esc(value) {
  return String(value === null || value === undefined ? "" : value)
    .replace(/[&<>"']/g, (c) => ESC_MAP[c]);
}

// An element. attrs:
// - onX: a listener for event "x" (onClick -> "click");
// - class or className: the class list, as one string;
// - style: a string, or an object assigned to el.style;
// - dataset: an object assigned to el.dataset;
// - anything else: an attribute, never parsed as markup; true makes it an
//   empty attribute, and null, undefined and false leave it out.
// Attributes set defaults on new elements (value, checked, selected), which
// is what they show; a <select>'s value is set once its options exist.
// Children may be nodes, strings (as text) or arrays of them; null,
// undefined and false are skipped.
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (/^on[A-Z]/.test(name) && typeof value === "function") {
      el.addEventListener(name.slice(2).toLowerCase(), value);
    } else if (name === "class" || name === "className") {
      el.className = value;
    } else if (name === "style" && typeof value === "object") {
      Object.assign(el.style, value);
    } else if (name === "dataset") {
      Object.assign(el.dataset, value);
    } else {
      el.setAttribute(name, value === true ? "" : String(value));
    }
  }
  appendChildren(el, children);
  return el;
}

function appendChildren(el, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    if (Array.isArray(child)) appendChildren(el, child);
    else el.append(child instanceof Node ? child : String(child));
  }
}

// Help text that shows one line and keeps the rest a click away: ``short``,
// then ``more`` (text, nodes or an array of them) under an "i" beside it, the
// markup the templates write (css/dark.css details.more). No ``more``: the
// line alone. The "i" is held to the line's last word, not wrapped onto a
// line of its own.
export function hint(short, more) {
  if (!more || (Array.isArray(more) && !more.length)) return [short];
  return [short, "\u00a0", h("details", { class: "more" }, h("summary", { "aria-label": "More" }, "i"), h("span", { class: "more-body" }, more))];
}

// A hint's "i" titled with the rest of its hint while it is closed, so a mouse
// resting on it shows the rest as a tooltip; open, the rest is in view and the
// title is dropped. A click or a tap still opens it in place, the one way a
// touch screen has.
export function titleHint(summary) {
  const details = summary.parentElement;
  const body = details && details.querySelector(".more-body");
  if (!body) return;
  summary.title = details.open ? "" : body.textContent.replace(/\s+/g, " ").trim();
}

// Every hint under ``root`` gets its tooltip as the mouse comes onto its "i":
// the templates' and the scripts' alike, whenever they were written.
export function tooltipHints(root = document) {
  root.addEventListener("mouseover", (event) => {
    const summary = event.target instanceof Element ? event.target.closest("details.more > summary") : null;
    if (summary) titleHint(summary);
  });
}

// ``el``'s content made hint()'s.
export function setHint(el, short, more) {
  el.replaceChildren(...hint(short, more));
}

const savedLabels = new WeakMap();

// Disable a button while its request runs. With a busyLabel, the label is a
// spinner and that text meanwhile (calling again changes the text); busy =
// false puts back the label it had before the first call, and enables it.
export function setBusy(button, busy, busyLabel) {
  if (busy) {
    if (!savedLabels.has(button)) savedLabels.set(button, Array.from(button.childNodes));
    button.disabled = true;
    if (busyLabel !== undefined && busyLabel !== null) {
      button.replaceChildren(h("span", { class: "spinner-border spinner-border-sm" }), " " + busyLabel);
    }
    return;
  }
  const label = savedLabels.get(button);
  if (label) {
    button.replaceChildren(...label);
    savedLabels.delete(button);
  }
  button.disabled = false;
}
