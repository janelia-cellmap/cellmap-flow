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
// row the answer gives nothing for is hidden, and the status line says why
// a publish has no browser form. A model published before shows where it
// went when the panel opens.
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
// is signed out there, not here. A pasted token goes to the credentials
// route at once and the field is emptied once it is taken: the route keeps
// it in memory (the page's, or the dashboard's process's), never in the
// page's storage; nothing here keeps it, logs it or puts it in a message.
import { HUGGINGFACE_SIGN_IN_EVENT, listen } from "../../lib/events.js";
import { formatMb } from "../../lib/format.js";
import { poll } from "../../lib/poll.js";
import { getAnswer, postAnswer } from "./requests.js";

const CREDENTIALS = "/api/huggingface/credentials";

// What a running publish is doing, by the step the route answers.
const STEP_WORDS = {
  exporting: "Exporting the model (a cluster job)",
  creating: "Making the repository",
  preparing: "Preparing the upload",
  hashing: "Hashing",
  uploading: "Uploading",
  committing: "Committing",
  checking: "Checking what Hugging Face has",
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
  let poller = null;
  let deviceWatch = null;  // the credentials polled while a sign-in with a code waits

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
      if (followed && followed.status === "failed") say(followed.error);
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
    running = answer.status === "running" || answer.small_status === "running";
    publishBtn.disabled = running || !signedIn();
    if (answer.status === "running") {
      const fraction = typeof answer.fraction === "number" ? `: ${Math.round(100 * answer.fraction)}%` : "";
      say(`${STEP_WORDS[answer.step] || "Publishing"}${answer.file ? ` ${answer.file}` : ""}${fraction}…`);
      startPolling();
      return;
    }
    // The smaller copy follows the publish by hours: asked on until it is in.
    if (answer.small_status === "running") startPolling();
    else stopPolling();
    if (answer.status === "failed") say(answer.error);
    if (answer.status === "done") {
      const how = answer.created === true ? " (a new repository)" : answer.created === false ? " (a new commit)" : "";
      // What Hugging Face says, read back after the commit: a private
      // repository's links open it for no one, this app reading public ones only.
      const seen = answer.private === true
        ? " Hugging Face shows every file. The repository is private, so its links will not open it: make it public in its Settings on Hugging Face."
        : answer.private === false ? " Hugging Face shows every file, in a public repository." : "";
      // A server's model the browser cannot run (too big for an .onnx file,
      // a 2D model) is published for servers alone, and the export says why.
      const browser = answer.no_browser ? ` It has no copy for the browser. ${answer.no_browser.replace(/\.?$/, ".")}` : "";
      say(`Published to ${answer.repo}${how}${answer.at ? `, ${new Date(answer.at).toLocaleString()}` : ""}.${seen}${browser}${smallLine(answer)}`);
    }
    showResult(answer.status === "done" ? answer : null);
  }

  // Where the smaller copy has got, after the model's own publish.
  function smallLine(answer) {
    if (answer.small_status === "running") {
      return answer.small_step === "distilling"
        ? " The smaller copy is training on the cluster (it can take hours); this page can stay closed meanwhile, but the dashboard must keep running."
        : ` The smaller copy: ${STEP_WORDS[answer.small_step] || "publishing"}${answer.small_file ? ` ${answer.small_file}` : ""}…`;
    }
    if (answer.small_status === "failed") return ` The smaller copy was not published: ${answer.small_error}`;
    if (answer.small_status === "done") return " The smaller copy is in the repository too.";
    return "";
  }

  // The refs and links of a publish that is done, each row only where the
  // answer gives one (a browser's publish has no server ref, a server's
  // without a browser form no folder ref or links, and either without a
  // dataset no dataset link); or none.
  function showResult(answer) {
    result.hidden = !answer;
    for (const [key, el] of Object.entries(fields)) {
      el.textContent = (answer && answer[key]) || "";
      if (el.tagName === "A") el.href = el.textContent || "#";
      el.parentElement.hidden = !el.textContent;
    }
  }

  // A running publish's progress, every second, until it is done or failed.
  function startPolling() {
    if (poller) return;
    const name = model;
    poller = poll(async ({ stale }) => {
      const answer = await getAnswer(publishUrl(name));
      if (stale() || name !== model) return false;
      showPublish(answer);
      return answer.status === "running" || answer.small_status === "running";
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

  // A popup, where the answer gives its URL: opened in the click itself,
  // so no popup blocker stops it; it hands its token to the page, which
  // says so (HUGGINGFACE_SIGN_IN_EVENT). Else a code, which the route asks
  // Hugging Face for and then waits on (showDevice).
  signInBtn.addEventListener("click", async () => {
    if (account && account.oauth_url) {
      const popup = window.open(account.oauth_url, "cellmap-flow-huggingface-sign-in", "popup,width=520,height=720");
      say(popup ? "Signing in in the Hugging Face window…" : "The browser blocked the sign-in window: allow pop-ups for this page, or paste an access token.");
      return;
    }
    if (!account || !account.device_sign_in) return;
    say("Asking Hugging Face for a sign-in code…");
    try {
      const answer = await postAnswer(CREDENTIALS, { device: true });
      say(answer.success ? "" : answer.error);
      if (answer.success) showAccount(answer);
    } catch (e) {
      say(`Could not reach the page's backend: ${e.message}`);
    }
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
    const remember = account && account.can_remember ? { remember: rememberBox.checked } : {};
    const answer = await postAnswer(CREDENTIALS, { token, ...remember });
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
        ...(loraBox ? { lora_copy: loraBox.checked } : {}),
        ...(smallBox ? { small_copy: smallBox.checked } : {}),
      }));
    } catch (e) {
      say(`Could not start the publish: ${e.message}`);
    }
    if (!running) publishBtn.disabled = !signedIn();
  });

  for (const copy of panel.querySelectorAll("[data-copy]")) {
    copy.addEventListener("click", async () => {
      const el = document.getElementById(copy.dataset.copy);
      try {
        await navigator.clipboard.writeText(el.textContent);
        copy.textContent = "Copied";
        setTimeout(() => { copy.textContent = "Copy"; }, 1500);
      } catch (_) {
        // The clipboard refused (a page the browser does not trust with
        // it): selected, for Ctrl+C.
        window.getSelection().selectAllChildren(el);
      }
    });
  }

  return { showFor };
}
