// The builder's toasts: a message in the bottom-right corner for 5 s, stacked
// above the ones still showing. type is "success", "error" or "info".
export function showMessage(msg, type) {
  const div = document.createElement("div");
  div.className = `message ${type}`;
  div.textContent = msg;
  document.body.appendChild(div);

  let bottomPosition = 20;
  const messages = document.querySelectorAll(".message");
  messages.forEach((m, idx) => {
    if (idx < messages.length - 1) {  // every one but the new one
      bottomPosition += m.offsetHeight + 10;
    }
  });
  div.style.bottom = bottomPosition + "px";

  setTimeout(() => div.remove(), 5000);
}
