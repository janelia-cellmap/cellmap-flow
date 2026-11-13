// The builder's dialogs (the .modal elements) and the dark backdrop behind
// them.
//
// Each dialog is registered once, with what closing it must also do (stop a
// poll, forget its state). Its [data-close] buttons (✕, Cancel, Close) close
// it, and so does a click on the backdrop -- except for the bounding-box
// drawing viewer: closing that drops the boxes drawn so far, so it takes its
// own Cancel button.
const dialogs = new Map();  // id -> {onClose, backdropCloses}

export function registerDialog(id, { onClose = () => {}, backdropCloses = true } = {}) {
  dialogs.set(id, { onClose, backdropCloses });
  document.getElementById(id).querySelectorAll("[data-close]").forEach((button) => {
    button.addEventListener("click", () => closeDialog(id));
  });
}

export function openDialog(id) {
  document.getElementById("modal-overlay").style.display = "block";
  document.getElementById(id).style.display = "block";
}

export function closeDialog(id) {
  document.getElementById("modal-overlay").style.display = "none";
  document.getElementById(id).style.display = "none";
  dialogs.get(id).onClose();
}

export function initDialogs() {
  document.getElementById("modal-overlay").addEventListener("click", (event) => {
    if (event.target !== event.currentTarget) return;
    dialogs.forEach(({ backdropCloses }, id) => {
      if (backdropCloses && document.getElementById(id).style.display !== "none") closeDialog(id);
    });
  });
}
