// Copy buttons: a button with data-copy="<id>" copies the text of the
// element with that id.
//
// navigator.clipboard is only there in a secure context (https, or
// localhost), and a dashboard on a cluster node is opened over plain http at
// the node's address, where it is undefined. There the text is copied as a
// selection is, from a read-only field made for it (document.execCommand,
// which browsers still honour in a click). Where neither works, the text is
// left selected and the button says to press Ctrl+C (Cmd+C on a Mac).

// Whether `text` went to the clipboard.
export async function copyText(text) {
  if (navigator.clipboard) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (_) {
      // Refused (the page is not focused, or not trusted with it): as below.
    }
  }
  const field = document.createElement("textarea");
  field.value = text;
  field.readOnly = true;
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.appendChild(field);
  const focused = document.activeElement;
  field.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch (_) {
    copied = false;
  }
  field.remove();
  if (focused && focused.focus) focused.focus({ preventScroll: true });
  return copied;
}

// Each [data-copy] button inside `root`: Copy, then "Copied" for a moment,
// or the text selected and how to copy it by hand.
export function initCopyButtons(root) {
  const keys = /Mac|iPhone|iPad/.test(navigator.platform) ? "Cmd+C" : "Ctrl+C";
  for (const button of root.querySelectorAll("[data-copy]")) {
    const label = button.textContent;
    let reset = null;
    button.addEventListener("click", async () => {
      const el = document.getElementById(button.dataset.copy);
      const copied = await copyText(el.textContent);
      if (!copied) window.getSelection().selectAllChildren(el);
      button.textContent = copied ? "Copied" : `Selected: press ${keys}`;
      clearTimeout(reset);
      reset = setTimeout(() => { button.textContent = label; }, copied ? 1500 : 4000);
    });
  }
}
