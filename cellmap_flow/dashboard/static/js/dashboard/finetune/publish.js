// Publish to Hugging Face, on the Training Status card, in both modes, for
// the model a job's status says is served (job-monitor.js shows it then; in
// the browser beside Download finetuned model). The button opens the panel,
// which says who is signed in to Hugging Face and offers Sign in or a
// pasted access token; then the repository (by default the user's own,
// named after the model), whether a new one is private, and whether the
// model card names the dataset. Publish starts the upload (on a server, the
// export first: TorchScript for servers, and ONNX for the browser where it
// can make one), whose progress is polled, and the result is the model's
// Hugging Face page, its refs and the two links that open it in the app,
// the model alone and on this dataset, each with a Copy button. A server's
// publish has a ref for servers, the repository's (its TorchScript), and,
// with a browser form, the folder's that both modes open and the links; a
// row the answer gives nothing for is hidden, and the progress says why a
// publish has no browser form. A model published before shows where it
// went when the panel opens.
//
// The progress, and once done the links, are in a box of their own below
// the panel, shown whether the panel is open or not, until Hide: the step
// the publish is on, the cluster job it waits on (its id, whether it is
// still queued; its log in the line's tooltip), each file as it is
// uploaded, and the smaller copy's training, epoch by epoch, then its
// upload. The answers are polled once at a time, while the page is shown,
// and only what changed is rewritten, so that a selection in the box stays.
// A publish running when the page loads shows at once.
//
// The routes are web/src/backend/routes/publish.ts's in the browser and
// dashboard/routes/finetune/publish.py's on a server, and answer alike.
// Where the two differ the credentials answer says so, and the panel goes
// by it, not by the mode: Sign in opens a popup where it gives an
// `oauth_url` (the browser's), and asks for a
// code to enter at huggingface.co where it offers `device_sign_in` (the
// server's), followed until the sign-in is done or has failed (`device`);
// Remember on this server is offered where it says `can_remember`; and a
// token it says is "saved" (on the server, by `hf auth login` or Remember)
// is signed out there, not here. What Sign in and a pasted token come to is
// said under the account line, where they are; one that fails puts the
// token field in focus, the other way to sign in, and once signed in the
// Publish button is brought into view and focused. A pasted token goes to
// the credentials route at once and the field is emptied once it is taken:
// the route keeps it in memory (the page's, or the dashboard's process's),
// never in the page's storage; nothing here keeps it, logs it or puts it in
// a message.
import { initCopyButtons } from "../../lib/clipboard.js";
import { HUGGINGFACE_SIGN_IN_EVENT, listen } from "../../lib/events.js";
import { formatMb } from "../../lib/format.js";
import { poll } from "../../lib/poll.js";
import { getAnswer, postAnswer } from "./requests.js";

const CREDENTIALS = "/api/huggingface/credentials";

// What a running publish is doing, by the step the route answers.
const STEP_WORDS = {
  exporting: "Exporting the model on the cluster",
  creating: "Making the repository",
  preparing: "Preparing the upload",
  hashing: "Hashing",
  uploading: "Uploading",
  committing: "Committing",
  checking: "Checking what Hugging Face has",
};

// Text only where it changed: the box is redrawn on every poll, and the same
// text written again would end a selection in it.
function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

// The cluster job a step waits on, as the answers' `job` gives it (a
// server's: its LSF id, and whether it still waits in the queue); nothing
// for a job without an id (a process on the dashboard's machine).
function jobWords(job) {
  if (!job || !job.id) return "";
  return ` (LSF job ${job.id}${job.queued ? ", waiting in the cluster's queue" : ""})`;
}

export function initPublish() {
  const button = document.getElementById("publishFinetunedBtn");
  const panel = document.getElementById("publishPanel");
  const accountLine = document.getElementById("hfAccount");
  const signInBtn = document.getElementById("hfSignInBtn");
  const signOutBtn = document.getElementById("hfSignOutBtn");
  const signInLine = document.getElementById("hfSignInStatus");
  const tokenFields = document.getElementById("hfTokenFields");
  const tokenHint = document.getElementById("hfTokenHint");
  const tokenInput = document.getElementById("hfToken");
  const rememberField = document.getElementById("hfRememberField");
  const rememberBox = document.getElementById("hfRemember");
  const deviceBox = document.getElementById("hfDevice");
  const deviceCode = document.getElementById("hfDeviceCode");
  const deviceLink = document.getElementById("hfDeviceLink");
  const repoInput = document.getElementById("publishRepo");
  const privateBox = document.getElementById("publishPrivate");
  const datasetBox = document.getElementById("publishIncludeDataset");
  // The server's only (its publish uploads a copy of the network; the template leaves it out elsewhere).
  const loraBox = document.getElementById("publishLoraCopy");
  const loraCost = document.getElementById("publishLoraCost");
  const loraCostAnyModel = loraCost ? loraCost.textContent : "";
  // The server's only too: a smaller copy, trained on the cluster.
  const smallBox = document.getElementById("publishSmallCopy");
  const smallCost = document.getElementById("publishSmallCost");
  const smallCostAnyModel = smallCost ? smallCost.textContent : "";
  const publishBtn = document.getElementById("publishBtn");
  const statusLine = document.getElementById("publishStatus");
  const progress = document.getElementById("publishProgress");
  const progressRepo = document.getElementById("publishProgressRepo");
  const progressClose = document.getElementById("publishProgressClose");
  const stageLine = document.getElementById("publishStage");
  const smallStageLine = document.getElementById("publishSmallStage");
  const result = document.getElementById("publishResult");
  const fields = {
    url: document.getElementById("publishUrl"),
    server_ref: document.getElementById("publishServerRef"),
    ref: document.getElementById("publishRef"),
    try_link: document.getElementById("publishTryLink"),
    dataset_link: document.getElementById("publishDatasetLink"),
    small_ref: document.getElementById("publishSmallRef"),
    small_try_link: document.getElementById("publishSmallTryLink"),
  };

  let model = null;  // the model the panel publishes (the job's)
  let account = null;  // the credentials route's last answer
  let defaultRepo = "";  // the repository field's default, replaced while the user has not changed it
  let running = false;
  let closed = false;  // the progress box hidden with Hide, until the panel opens or a publish starts
  let poller = null;
  let deviceWatch = null;  // the credentials polled while a sign-in with a code waits

  const publishUrl = (name) => `/api/finetune/model/${encodeURIComponent(name)}/publish`;
  const say = (text) => setText(statusLine, text || "");
  const saySignIn = (text) => {
    setText(signInLine, text || "");
    signInLine.hidden = !text;
  };

  // Shown with the Download link, for `name`; hidden with it (null).
  function showFor(name) {
    if (name === model) return;
    model = name || null;
    stopPolling();
    running = false;
    closed = false;
    button.style.display = model ? "inline-block" : "none";
    showProgress(null);
    say("");
    if (!model) {
      panel.hidden = true;
      button.setAttribute("aria-expanded", "false");
      return;
    }
    if (!panel.hidden) refresh();
    else loadRunning();
  }

  async function refresh() {
    const name = model;
    try {
      showAccount(await getAnswer(CREDENTIALS));
      const answer = name && await getAnswer(publishUrl(name));
      if (answer && name === model) showPublish(answer);
    } catch (e) {
      say(`Could not reach the page's backend: ${e.message}`);
    }
  }

  // With the panel closed, a publish of the model under way in the backend
  // (started before the page was loaded) shows its progress all the same.
  async function loadRunning() {
    const name = model;
    try {
      const answer = await getAnswer(publishUrl(name));
      if (name === model && answer.success && (answer.status === "running" || answer.small_status === "running")) {
        showPublish(answer);
      }
    } catch (_) {
      // Shown when the panel is opened, which asks again.
    }
  }

  function signedIn() {
    return Boolean(account && account.signed_in);
  }

  function showAccount(answer) {
    if (!answer.success) {
      saySignIn(answer.error);
      return;
    }
    account = answer;
    const yes = signedIn();
    const how = {
      token: " (access token)",
      saved: " (saved on this server; run hf auth logout there to sign out)",
    }[account.via] || "";
    accountLine.textContent = yes ? `Signed in to Hugging Face as ${account.user}${how}` : "Not signed in to Hugging Face";
    const waiting = !yes && account.device && account.device.status === "pending" ? account.device : null;
    signInBtn.hidden = yes || Boolean(waiting) || !(account.oauth_url || account.device_sign_in);
    signOutBtn.hidden = !yes || account.via === "saved";
    showDevice(waiting);
    tokenFields.hidden = yes;
    tokenHint.hidden = yes;
    rememberField.hidden = yes || !account.can_remember;
    const fallback = yes && model ? `${account.user}/${model}` : "";
    if (!repoInput.value.trim() || repoInput.value.trim() === defaultRepo) repoInput.value = fallback;
    defaultRepo = fallback;
    publishBtn.disabled = !yes || running;
  }

  // Signed in, by a token or a sign-in: the token fields are gone, and the
  // next step is the Publish button, brought into view.
  function showNextStep() {
    publishBtn.scrollIntoView({ block: "nearest" });
    publishBtn.focus({ preventScroll: true });
  }

  // A sign-in failed: the other way in is the token field.
  function signInFailed(why) {
    saySignIn(why);
    if (!tokenFields.hidden) tokenInput.focus();
  }

  // A sign-in with a code that waits for the user: the code, and the link
  // to enter it at (Hugging Face's own with the code in it, where it gives
  // one), followed until the sign-in is done or has failed; or none.
  function showDevice(device) {
    deviceBox.hidden = !device;
    if (!device) return;
    deviceCode.textContent = device.user_code;
    deviceLink.href = device.verification_uri_complete || device.verification_uri;
    deviceLink.textContent = device.verification_uri;
    if (deviceWatch && deviceWatch.running) return;
    deviceWatch = poll(async ({ stale }) => {
      const answer = await getAnswer(CREDENTIALS);
      if (stale()) return false;
      showAccount(answer);
      const followed = answer.success ? answer.device : null;
      if (followed && followed.status === "failed") signInFailed(followed.error);
      if (signedIn()) showNextStep();
      return Boolean(followed && followed.status === "pending");
    }, { intervalMs: 2000, immediate: false });
  }

  function showPublish(answer) {
    if (!answer.success) {
      say(answer.error);
      return;
    }
    // What model.pt2 costs for this model, where its training said how big
    // its network is (the server's status answer); else the rule for any.
    if (loraCost && "lora_copy_bytes" in answer) {
      loraCost.textContent = answer.lora_copy_bytes
        ? `Adds ${formatMb(answer.lora_copy_bytes / 1e6)} for this model: a second copy of its weights, in the upload and on Hugging Face.`
        : loraCostAnyModel;
    }
    if (smallCost && "small_copy_bytes" in answer) {
      smallCost.textContent = answer.small_copy_bytes
        ? `Trains it on one cluster GPU, for up to ${answer.small_copy_hours.pruned} hours when the model is a funlib UNet (it starts from the model's own channels) and ${answer.small_copy_hours.random} otherwise (from scratch), then adds it to the repository: about ${formatMb(answer.small_copy_bytes / 1e6)} in its four files for this model.`
        : smallCostAnyModel;
    }
    // The smaller copy follows the publish by hours: asked on until it is in.
    running = answer.status === "running" || answer.small_status === "running";
    publishBtn.disabled = running || !signedIn();
    showProgress(answer.status === "none" ? null : answer);
    if (running) startPolling();
    else stopPolling();
  }

  // The progress box: the repository, where the model has got, where its
  // smaller copy has, and once the model is published its links; hidden
  // for no publish, and after Hide.
  function showProgress(answer) {
    progress.hidden = !answer || closed;
    if (!answer) {
      showResult(null);
      return;
    }
    setText(progressRepo, answer.repo || "");
    setText(stageLine, modelLine(answer));
    stageLine.title = answer.job && answer.job.log ? `Its log: ${answer.job.log}` : "";
    const small = smallLine(answer);
    setText(smallStageLine, small);
    smallStageLine.hidden = !small;
    smallStageLine.title = answer.small_job && answer.small_job.log ? `Its log: ${answer.small_job.log}` : "";
    showResult(answer.status === "done" ? answer : null);
  }

  // Where the model's own publish has got.
  function modelLine(answer) {
    if (answer.status === "running") {
      const fraction = typeof answer.fraction === "number" ? `: ${Math.round(100 * answer.fraction)}%` : "";
      return `${STEP_WORDS[answer.step] || "Publishing"}${answer.file ? ` ${answer.file}` : ""}${fraction}${jobWords(answer.job)}…`;
    }
    if (answer.status === "failed") {
      const log = answer.job && answer.job.log ? ` Its log: ${answer.job.log}` : "";
      return `Not published: ${answer.error}${log}`;
    }
    const how = answer.created === true ? " (a new repository)" : answer.created === false ? " (a new commit)" : "";
    // What Hugging Face says, read back after the commit: a private
    // repository's links open it for no one, this app reading public ones only.
    const seen = answer.private === true
      ? " Hugging Face shows every file. The repository is private, so its links will not open it: make it public in its Settings on Hugging Face."
      : answer.private === false ? " Hugging Face shows every file, in a public repository." : "";
    // A server's model the browser cannot run (too big for an .onnx file,
    // a 2D model) is published for servers alone, and the export says why.
    // The smaller copy, where one was asked for, is then the browser's.
    const browser = !answer.no_browser ? "" : ` It runs on servers only. ${answer.no_browser.replace(/\.?$/, ".")}` +
      (answer.small_status ? " The smaller copy, below, is the one for the browser." : "");
    return `Published${how}${answer.at ? `, ${new Date(answer.at).toLocaleString()}` : ""}.${seen}${browser}`;
  }

  // Where the smaller copy has got: trained on the cluster while the model
  // is published, then added to the repository; for a model with no browser
  // form, the only copy the browser can run. Nothing when none was asked for.
  function smallLine(answer) {
    const status = answer.small_status;
    if (!status) return "";
    if (status === "failed") return `The smaller copy was not published: ${answer.small_error}`;
    if (status === "done") return "The smaller copy, for the browser, is in the repository too: its links are below.";
    const job = answer.small_job;
    if (answer.small_step !== "distilling") {
      const step = (STEP_WORDS[answer.small_step] || "Publishing").toLowerCase();
      return `The smaller copy for the browser: ${step}${answer.small_file ? ` ${answer.small_file}` : ""}…`;
    }
    const epoch = job && job.epoch ? `, epoch ${job.epoch} of ${job.epochs}` : "";
    return `The smaller copy for the browser is ${job && job.queued ? "to be trained" : "training"} on the cluster${epoch}${jobWords(job)}. ` +
      "It is uploaded when its training ends, which can take hours, and its links then show here; " +
      "the dashboard must keep running until then.";
  }

  // The refs and links of a publish that is done, each row only where the
  // answer gives one (a browser's publish has no server ref, a server's
  // without a browser form no folder ref or links, and either without a
  // dataset no dataset link); or none.
  function showResult(answer) {
    result.hidden = !answer;
    for (const [key, el] of Object.entries(fields)) {
      const text = (answer && answer[key]) || "";
      setText(el, text);
      if (el.tagName === "A" && el.getAttribute("href") !== (text || "#")) el.href = text || "#";
      el.parentElement.hidden = !text;
    }
  }

  // A running publish's progress, every second while the page is shown,
  // until it is done or failed.
  function startPolling() {
    if (poller) return;
    const name = model;
    poller = poll(async ({ stale }) => {
      const answer = await getAnswer(publishUrl(name));
      if (stale() || name !== model) return false;
      showPublish(answer);
      return answer.status === "running" || answer.small_status === "running";
    }, { intervalMs: 1000, immediate: false, pauseWhenHidden: true });
  }

  function stopPolling() {
    if (poller) poller.stop();
    poller = null;
  }

  button.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    button.setAttribute("aria-expanded", String(!panel.hidden));
    if (panel.hidden) return;
    closed = false;
    refresh();
  });

  progressClose.addEventListener("click", () => {
    closed = true;
    progress.hidden = true;
  });

  // A popup, where the answer gives its URL: opened in the click itself,
  // so no popup blocker stops it; it hands its token to the page, which
  // says so (HUGGINGFACE_SIGN_IN_EVENT). Else a code, which the route asks
  // Hugging Face for and then waits on (showDevice).
  signInBtn.addEventListener("click", async () => {
    if (account && account.oauth_url) {
      const popup = window.open(account.oauth_url, "cellmap-flow-huggingface-sign-in", "popup,width=520,height=720");
      if (popup) saySignIn("Signing in in the Hugging Face window…");
      else signInFailed("The browser blocked the sign-in window: allow pop-ups for this page, or paste an access token.");
      return;
    }
    if (!account || !account.device_sign_in) return;
    saySignIn("Asking Hugging Face for a sign-in code…");
    signInBtn.disabled = true;
    try {
      const answer = await postAnswer(CREDENTIALS, { device: true });
      if (answer.success) {
        saySignIn("");
        showAccount(answer);
      } else {
        signInFailed(answer.error);
      }
    } catch (e) {
      signInFailed(`Could not reach the page's backend: ${e.message}`);
    }
    signInBtn.disabled = false;
  });
  listen(HUGGINGFACE_SIGN_IN_EVENT, async (event) => {
    const detail = event.detail || {};
    if (detail.error) signInFailed(detail.error);
    else saySignIn("");
    await refresh();
    if (signedIn()) showNextStep();
  });

  signOutBtn.addEventListener("click", async () => {
    showAccount(await postAnswer(CREDENTIALS, { forget: true }));
    saySignIn("");
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
    const remember = account && account.can_remember ? { remember: rememberBox.checked } : {};
    let answer;
    try {
      answer = await postAnswer(CREDENTIALS, { token, ...remember });
    } catch (e) {
      sending = null;
      saySignIn(`Could not reach the page's backend: ${e.message}`);
      return;
    }
    if (!answer.success) {
      saySignIn(answer.error);
      return;
    }
    tokenInput.value = "";
    sending = null;
    saySignIn("");
    showAccount(answer);
    showNextStep();
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
      const answer = await postAnswer(publishUrl(model), {
        repo: repoInput.value.trim(),
        private: privateBox.checked,
        include_dataset: datasetBox.checked,
        ...(loraBox ? { lora_copy: loraBox.checked } : {}),
        ...(smallBox ? { small_copy: smallBox.checked } : {}),
      });
      if (answer.success) {
        say("");
        closed = false;
      }
      showPublish(answer);
      if (answer.success) progress.scrollIntoView({ block: "nearest" });
    } catch (e) {
      say(`Could not start the publish: ${e.message}`);
    }
    if (!running) publishBtn.disabled = !signedIn();
  });

  initCopyButtons(progress);

  return { showFor };
}
