// Browser mode: Publish to Hugging Face, on the Training Status card beside
// Download finetuned model, for the same model (job-monitor.js shows both
// for the model a job's status says is served). The button opens the panel,
// which says who is signed in to Hugging Face and offers Sign in (a popup,
// when the app has an OAuth app) or a pasted access token; then the
// repository (by default the user's own, named after the model), whether a
// new one is private, and whether the model card names the dataset. Publish
// starts the upload, whose progress is polled, and the result is the
// model's Hugging Face page, its ref and the two links that open it in the
// app, the model alone and on this dataset, each with a Copy button. A model
// published before shows where it went when the panel opens.
//
// The routes are web/src/backend/routes/publish.ts's. A pasted token goes
// to the credentials route at once and the field is emptied once it is
// taken: the route keeps it in the page's memory, never in storage; nothing
// here keeps it, logs it or puts it in a message.
import { HUGGINGFACE_SIGN_IN_EVENT, listen } from "../../lib/events.js";
import { poll } from "../../lib/poll.js";
import { getAnswer, postAnswer } from "./requests.js";

const CREDENTIALS = "/api/huggingface/credentials";

// What a running publish is doing, by the step the route answers.
const STEP_WORDS = {
  creating: "Making the repository",
  preparing: "Preparing the upload",
  hashing: "Hashing",
  uploading: "Uploading",
  committing: "Committing",
};

export function initPublish() {
  const button = document.getElementById("publishFinetunedBtn");
  const panel = document.getElementById("publishPanel");
  const accountLine = document.getElementById("hfAccount");
  const signInBtn = document.getElementById("hfSignInBtn");
  const signOutBtn = document.getElementById("hfSignOutBtn");
  const tokenFields = document.getElementById("hfTokenFields");
  const tokenHint = document.getElementById("hfTokenHint");
  const tokenInput = document.getElementById("hfToken");
  const repoInput = document.getElementById("publishRepo");
  const privateBox = document.getElementById("publishPrivate");
  const datasetBox = document.getElementById("publishIncludeDataset");
  const publishBtn = document.getElementById("publishBtn");
  const statusLine = document.getElementById("publishStatus");
  const result = document.getElementById("publishResult");
  const fields = {
    url: document.getElementById("publishUrl"),
    ref: document.getElementById("publishRef"),
    try_link: document.getElementById("publishTryLink"),
    dataset_link: document.getElementById("publishDatasetLink"),
  };

  let model = null;  // the model the panel publishes (the job's)
  let account = null;  // the credentials route's last answer
  let defaultRepo = "";  // the repository field's default, replaced while the user has not changed it
  let running = false;
  let poller = null;

  const publishUrl = (name) => `/api/finetune/model/${encodeURIComponent(name)}/publish`;
  const say = (text) => { statusLine.textContent = text || ""; };

  // Shown with the Download link, for `name`; hidden with it (null).
  function showFor(name) {
    if (name === model) return;
    model = name || null;
    stopPolling();
    running = false;
    button.style.display = model ? "inline-block" : "none";
    showResult(null);
    say("");
    if (!model) {
      panel.hidden = true;
      button.setAttribute("aria-expanded", "false");
      return;
    }
    if (!panel.hidden) refresh();
  }

  async function refresh() {
    try {
      showAccount(await getAnswer(CREDENTIALS));
      if (model) showPublish(await getAnswer(publishUrl(model)));
    } catch (e) {
      say(`Could not reach the page's backend: ${e.message}`);
    }
  }

  function signedIn() {
    return Boolean(account && account.signed_in);
  }

  function showAccount(answer) {
    if (!answer.success) {
      say(answer.error);
      return;
    }
    account = answer;
    const yes = signedIn();
    accountLine.textContent = yes
      ? `Signed in to Hugging Face as ${account.user}${account.via === "token" ? " (access token)" : ""}`
      : "Not signed in to Hugging Face";
    signInBtn.hidden = yes || !account.oauth_url;
    signOutBtn.hidden = !yes;
    tokenFields.hidden = yes;
    tokenHint.hidden = yes;
    const fallback = yes && model ? `${account.user}/${model}` : "";
    if (!repoInput.value.trim() || repoInput.value.trim() === defaultRepo) repoInput.value = fallback;
    defaultRepo = fallback;
    publishBtn.disabled = !yes || running;
  }

  function showPublish(answer) {
    if (!answer.success) {
      say(answer.error);
      return;
    }
    running = answer.status === "running";
    publishBtn.disabled = running || !signedIn();
    if (running) {
      const fraction = typeof answer.fraction === "number" ? `: ${Math.round(100 * answer.fraction)}%` : "";
      say(`${STEP_WORDS[answer.step] || "Publishing"}${answer.file ? ` ${answer.file}` : ""}${fraction}…`);
      startPolling();
      return;
    }
    stopPolling();
    if (answer.status === "failed") say(answer.error);
    if (answer.status === "done") {
      const how = answer.created === true ? " (a new repository)" : answer.created === false ? " (a new commit)" : "";
      say(`Published to ${answer.repo}${how}${answer.at ? `, ${new Date(answer.at).toLocaleString()}` : ""}.`);
    }
    showResult(answer.status === "done" ? answer : null);
  }

  // The links of a publish that is done, or none.
  function showResult(answer) {
    result.hidden = !answer;
    for (const [key, input] of Object.entries(fields)) input.value = (answer && answer[key]) || "";
    for (const el of result.querySelectorAll("[data-dataset-link]")) el.hidden = !(answer && answer.dataset_link);
  }

  // A running publish's progress, every second, until it is done or failed.
  function startPolling() {
    if (poller) return;
    const name = model;
    poller = poll(async ({ stale }) => {
      const answer = await getAnswer(publishUrl(name));
      if (stale() || name !== model) return false;
      showPublish(answer);
      return answer.status === "running";
    }, { intervalMs: 1000, immediate: false });
  }

  function stopPolling() {
    if (poller) poller.stop();
    poller = null;
  }

  button.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    button.setAttribute("aria-expanded", String(!panel.hidden));
    if (!panel.hidden) refresh();
  });

  // The popup is opened in the click itself, so no popup blocker stops it;
  // it hands its token to the page, which says so (HUGGINGFACE_SIGN_IN_EVENT).
  signInBtn.addEventListener("click", () => {
    if (!account || !account.oauth_url) return;
    const popup = window.open(account.oauth_url, "cellmap-flow-huggingface-sign-in", "popup,width=520,height=720");
    say(popup ? "Signing in in the Hugging Face window…" : "The browser blocked the sign-in window: allow pop-ups for this page, or paste an access token.");
  });
  listen(HUGGINGFACE_SIGN_IN_EVENT, (event) => {
    const detail = event.detail || {};
    say(detail.error || "");
    refresh();
  });

  signOutBtn.addEventListener("click", async () => {
    showAccount(await postAnswer(CREDENTIALS, { forget: true }));
    say("");
  });

  // A pasted token is handed over a moment after the field stops changing,
  // when it is left, or on Enter in it, and the field emptied once the route
  // has taken it.
  // The value being handed over, which is not sent twice (the field's
  // pause and its change both hand over), cleared with the field once the
  // route has taken it.
  let tokenTimer = null;
  let sending = null;
  async function handOverToken() {
    clearTimeout(tokenTimer);
    const token = tokenInput.value.trim();
    if (!token || token === sending) return;
    sending = token;
    const answer = await postAnswer(CREDENTIALS, { token });
    if (!answer.success) {
      say(answer.error);
      return;
    }
    tokenInput.value = "";
    sending = null;
    say("");
    showAccount(answer);
  }
  // Masked by CSS, which a password manager does not offer to save; a
  // browser that cannot mask a text field gets a password field (the AI
  // panel's credentials do the same, ai-annotate.js).
  if (!(window.CSS && CSS.supports("-webkit-text-security", "disc"))) tokenInput.type = "password";
  tokenInput.addEventListener("input", () => {
    clearTimeout(tokenTimer);
    tokenTimer = setTimeout(handOverToken, 500);
  });
  tokenInput.addEventListener("change", handOverToken);
  tokenInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    handOverToken();
  });

  publishBtn.addEventListener("click", async () => {
    if (!model) return;
    publishBtn.disabled = true;
    say("Starting…");
    try {
      showPublish(await postAnswer(publishUrl(model), {
        repo: repoInput.value.trim(),
        private: privateBox.checked,
        include_dataset: datasetBox.checked,
      }));
    } catch (e) {
      say(`Could not start the publish: ${e.message}`);
    }
    if (!running) publishBtn.disabled = !signedIn();
  });

  for (const copy of panel.querySelectorAll("[data-copy]")) {
    copy.addEventListener("click", async () => {
      const input = document.getElementById(copy.dataset.copy);
      try {
        await navigator.clipboard.writeText(input.value);
        copy.textContent = "Copied";
        setTimeout(() => { copy.textContent = "Copy"; }, 1500);
      } catch (_) {
        // The clipboard refused (a page the browser does not trust with
        // it): selected, for Ctrl+C.
        input.select();
      }
    });
  }

  return { showFor };
}
