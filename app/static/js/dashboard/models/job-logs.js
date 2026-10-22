// Inference job output (GET /api/job-logs).
//
// The dashboard's own log stream only carries what this process logs. A
// server that dies on startup, or 500s on every chunk, writes its traceback
// to the LSF job's output on whichever node it landed on -- so the only way
// to read it used to be ssh + bpeek.
import { getJSON } from "../../lib/api.js";
import { poll } from "../../lib/poll.js";

export const POLL_MS = 5000;

function render(data) {
  const area = document.getElementById("jobLogsArea");
  const statusEl = document.getElementById("jobLogsStatus");
  const jobs = (data && data.jobs) || [];
  if (!jobs.length) {
    area.value = "No inference jobs have been submitted yet.";
    statusEl.textContent = "";
    return;
  }
  area.value = jobs.map((j) => {
    const header = [
      j.model_name || "(unnamed)",
      j.job_id ? "job " + j.job_id : null,
      j.status || null,
      j.host || null,
    ].filter(Boolean).join("  |  ");
    // null log means there is no way to read this job's output at all,
    // which is worth saying plainly rather than showing as empty.
    const body = j.log === null || j.log === undefined
      ? "(no output available for this job)"
      : (j.log || "(no output yet)");
    return "=== " + header + " ===\n" + body;
  }).join("\n\n");
  // Keep the newest output in view.
  area.scrollTop = area.scrollHeight;
  statusEl.textContent = "updated " + new Date().toLocaleTimeString();
}

// Returns ``watchStartingJobs()``, for a Submit; ``look()``, one look at the
// jobs for the tab's poll, and ``watching()``, whether a Submit's jobs are
// still being watched for.
export function init() {
  let follower = null;
  // /api/job-logs runs bpeek per job, each with its own timeout, so a
  // refresh can outlast the 5s follow interval. One at a time, whether it
  // comes from Follow, the button or a Submit's watch: overlapping requests
  // pile onto a dashboard that is already the slow part. Resolves to the
  // answer, or to nothing when none was drawn.
  let busy = false;
  // After a Submit, show Job Logs and keep it current until no job is still
  // starting (waiting in a queue, or installing its environment), even with
  // Follow off: until then the page showed nothing of what Submit started.
  let watch = null;  // { looks }

  function refresh() {
    if (busy) return Promise.resolve();
    busy = true;
    const statusEl = document.getElementById("jobLogsStatus");
    return getJSON("/api/job-logs")
      .then((data) => { render(data); return data; })
      .catch((e) => { statusEl.textContent = "error: " + e; })
      .then((data) => { busy = false; return data; });
  }

  // The first looks can come before a job is handed to LSF at all, which
  // lists nothing: an empty list ends the watch only after 6 looks.
  function look() {
    const mine = watch;
    mine.looks += 1;
    if (mine.looks >= 60) watch = null;
    return refresh().then((data) => {
      if (!data || mine !== watch) return;
      const jobs = data.jobs || [];
      const starting = jobs.some((j) => j.status === "starting");
      if (!starting && !(jobs.length === 0 && mine.looks < 6)) watch = null;
    });
  }

  function watchStartingJobs() {
    const area = document.getElementById("jobLogsArea");
    if (area.style.display === "none") document.getElementById("jobLogsBtn").click();
    if (follower) return;
    watch = { looks: 0 };
    look();
  }

  document.getElementById("jobLogsBtn").addEventListener("click", function () {
    const area = document.getElementById("jobLogsArea");
    const showing = area.style.display !== "none";
    area.style.display = showing ? "none" : "block";
    this.textContent = showing ? "Show Job Logs" : "Hide Job Logs";
    if (!showing) refresh();
  });

  document.getElementById("jobLogsFollow").addEventListener("change", function () {
    if (follower) {
      follower.stop();
      follower = null;
    }
    if (this.checked) {
      const area = document.getElementById("jobLogsArea");
      if (area.style.display === "none") document.getElementById("jobLogsBtn").click();
      follower = poll(refresh, { intervalMs: POLL_MS, immediate: false });
      watch = null;  // Follow keeps it current
    }
  });

  return { watchStartingJobs, look, watching: () => watch !== null };
}
