// The Flow Logs panel under the canvas: the dashboard's log lines, streamed
// from /api/logs/stream as they are written. The stream is not reopened
// after an error; a reload does that.
function startLogStreaming() {
  const eventSource = new EventSource("/api/logs/stream");
  const logOutput = document.getElementById("log-output");
  eventSource.onmessage = (event) => {
    if (event.data) {
      const logLine = document.createElement("div");
      logLine.textContent = event.data;
      logOutput.appendChild(logLine);
      logOutput.parentElement.scrollTop = logOutput.parentElement.scrollHeight;  // follow the end
    }
  };
  eventSource.onerror = (error) => {
    console.error("Log streaming error:", error);
    eventSource.close();
  };
}

function toggleLogPanel() {
  const logContent = document.getElementById("log-content");
  logContent.style.display = logContent.style.display === "none" ? "flex" : "none";
  document.querySelector(".log-toggle-btn").classList.toggle("collapsed");
}

export function initLogPanel() {
  document.querySelector(".log-toggle-btn").addEventListener("click", toggleLogPanel);
  document.addEventListener("DOMContentLoaded", startLogStreaming);
}
