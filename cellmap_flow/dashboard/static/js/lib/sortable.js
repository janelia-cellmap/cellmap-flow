// Reorder a list's items by dragging them by their handle (HTML5 drag and
// drop).
//
// An item is draggable only while its handle is held, so the fields in it
// still take clicks and text selection. A drag that did not start on one of
// the list's own items (text or a file from elsewhere) moves nothing.
// While dragged, an item has the class "dragging" and half opacity; it is
// placed before the first item whose middle is below the pointer.
// onReorder(item) runs when a drag of an item ends.
export function makeSortable(listEl, { itemSelector, handleSelector, onReorder } = {}) {
  let dragged = null;

  function itemOf(target) {
    const item = target && target.closest ? target.closest(itemSelector) : null;
    return item && listEl.contains(item) ? item : null;
  }

  listEl.addEventListener("mousedown", (event) => {
    const handle = event.target.closest ? event.target.closest(handleSelector) : null;
    const item = handle && listEl.contains(handle) ? itemOf(handle) : null;
    if (item) item.setAttribute("draggable", "true");
  });

  document.addEventListener("mouseup", () => {
    listEl.querySelectorAll(`${itemSelector}[draggable]`).forEach((item) => item.removeAttribute("draggable"));
  });

  listEl.addEventListener("dragstart", (event) => {
    const item = itemOf(event.target);
    if (!item) return;
    dragged = item;
    item.style.opacity = "0.5";
    // itemAfter skips .dragging, so the item is placed relative to the
    // others rather than to itself.
    item.classList.add("dragging");
  });

  listEl.addEventListener("dragend", (event) => {
    const item = itemOf(event.target);
    if (item) {
      item.style.opacity = "1";
      item.classList.remove("dragging");
      item.removeAttribute("draggable");
    }
    const moved = dragged;
    dragged = null;
    if (moved && onReorder) onReorder(moved);
  });

  listEl.addEventListener("dragover", (event) => {
    if (!dragged) return;
    event.preventDefault();
    const after = itemAfter(listEl, itemSelector, event.clientY);
    if (after == null) {
      listEl.appendChild(dragged);
    } else {
      listEl.insertBefore(dragged, after);
    }
  });
}

function itemAfter(listEl, itemSelector, y) {
  const items = [...listEl.querySelectorAll(`${itemSelector}:not(.dragging)`)];
  return items.reduce((closest, child) => {
    const box = child.getBoundingClientRect();
    const offset = y - box.top - box.height / 2;
    return offset < 0 && offset > closest.offset ? { offset, element: child } : closest;
  }, { offset: Number.NEGATIVE_INFINITY }).element;
}
