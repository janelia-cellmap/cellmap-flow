"""Publishing a model finetuned on this server to Hugging Face, under the user's own account.

The Training Status card's Publish to Hugging Face panel
(static/js/dashboard/finetune/publish.js) is the browser mode's too, and so
are the routes and their answers: web/src/backend/routes/publish.ts answers
them in the page, and this module answers them here, refusing what that
refuses.

- GET ``/api/huggingface/credentials``: who is signed in to Hugging Face and
  how, and the sign-in with a code that is waiting for the user, if any;
- POST ``/api/huggingface/credentials``: ``{token, remember?}``, a pasted
  access token, saved on this server too with ``remember``; ``{device:
  true}``, Sign in with Hugging Face; ``{forget: true}``, signing out;
- POST ``/api/finetune/model/<name>/publish``: ``{repo?, private?,
  include_dataset?}``, a publish started;
- GET ``/api/finetune/model/<name>/publish``: how it is going, or where the
  model went.

``<name>`` is a model a finetune job serves: the job's
``finetuned_model_name``, its latest iteration's, which the job's status
answer names and the panel is shown for (job-monitor.js). Its serving YAML
says what the model is (the base model, and the finetune's weights on it: a
LoRA adapter, or the whole network's) and the dataset it was trained on;
the job says how long it trained and to what loss.

A publish is a job in a thread of the dashboard, its steps as the answers
name them: "exporting", the finetune folded into the base network and
exported to ONNX at the served block size, by a cluster job of its own
(finetune.export_web); "creating", the repository made unless it is there
already; "uploading" each file and "committing" them, in one commit; and
"checking", the repository read back, so that done means Hugging Face shows
each file at the size sent. The model card and the manifest are
portable.publish's, which the browser's publish writes too; the upload is
huggingface_hub's.

Who publishes is one of three tokens, the answers' ``via``:

- "saved": huggingface_hub's own (``get_token``: HF_TOKEN, or the file ``hf
  auth login`` writes), used when no other is signed in here. ``hf auth
  logout`` on the server signs it out, not the panel;
- "token": one pasted in the panel, kept in this process's memory only, and
  with Remember on this server saved to huggingface_hub's file
  (``huggingface_hub.login``), after which it is "saved";
- "oauth": Sign in with Hugging Face, by its device flow (RFC 8628): Hugging
  Face gives a code, the user enters it at huggingface.co in any browser,
  and the dashboard asks for the token until it is given. The page of a
  dashboard on a cluster node is opened through a tunnel or a proxy, at an
  address the app's client metadata cannot list as a redirect, so the
  browser mode's popup, which returns to the page, cannot work here. A sign-in's token
  expires, so it is kept in memory only and never saved.

Every token, and a sign-in's device code, is registered with
ai_annotate.secrets before anything can log it, since the log panel streams
the log to the page; no answer carries one.
"""

import logging
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import yaml
from flask import jsonify, request

import cellmap_flow
from cellmap_flow.ai_annotate.secrets import redact, register_secret
from cellmap_flow.dashboard.requests import FinetunePublish, HuggingFaceCredentials, parse
from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.portable.publish import (
    HF_OAUTH_CLIENT_ID,
    HF_OAUTH_SCOPES,
    HUB,
    PUBLIC_APP,
    READ_ONLY,
    SIGN_IN_FIRST,
    TOKEN_REFUSED,
    hub_refusal,
    publication_files,
    published_links,
    ref_of,
    repo_refusal,
)

logger = logging.getLogger(__name__)

UNREACHABLE = ("Hugging Face could not be reached from this server: check its connection to huggingface.co "
               "(a firewall or a proxy may stop it) and try again.")

EXPIRED = "The sign-in code expired before it was entered at huggingface.co: sign in again."

# Hugging Face's device flow (https://huggingface.co/docs/hub/oauth#device-code-oauth).
DEVICE_URL = f"{HUB}/oauth/device"
TOKEN_URL = f"{HUB}/oauth/token"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
# How much longer to wait between asks after a slow_down (RFC 8628 3.5).
SLOW_DOWN_SECONDS = 5

# How often a publish asks its export job how it is going.
EXPORT_POLL_SECONDS = 5


def _error(message, status):
    return jsonify({"success": False, "error": message}), status


def _start_thread(target, *args):
    """Run ``target(*args)`` in a daemon thread. Tests replace this to run it at once."""
    threading.Thread(target=target, args=args, daemon=True, name="huggingface").start()


def _now():
    return datetime.now(timezone.utc)


def _iso(when):
    """``when`` as the browser writes a time (Date.toISOString): UTC, to the millisecond, with a Z."""
    return when.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --- What Hugging Face said, in words that say what to do ------------------------


def _hub_status(e):
    """The HTTP status of huggingface_hub's error (or requests'), or None for one that is not an answer."""
    return getattr(getattr(e, "response", None), "status_code", None)


def _unreachable(e):
    """Whether ``e`` is a network failure rather than an answer: no status, and
    an error of the connection, the OS's or the HTTP library's (httpx or
    requests, by huggingface_hub's version)."""
    library = type(e).__module__.split(".")[0] in ("httpx", "requests", "urllib3")
    return _hub_status(e) is None and (library or isinstance(e, (ConnectionError, TimeoutError)))


def _hub_message(e):
    """Hugging Face's reason for ``e`` in a sentence: the Hub's own message
    where it gave one, without the URLs huggingface_hub adds (an upload's
    are presigned storage URLs), and with any secret redacted."""
    text = getattr(e, "server_message", None) or str(e) or type(e).__name__
    return redact(re.sub(r"\s*(?:for url:?\s*)?https?://\S+", "", text, flags=re.I).strip())


def _token_error(e):
    """Why a token could not be taken (whoami failed): publish.ts's tokenError."""
    if _unreachable(e):
        return UNREACHABLE
    if _hub_status(e) == 401:
        return TOKEN_REFUSED
    return f"Hugging Face could not say whose token this is: {_hub_message(e)}"


def _publish_error(e, repo, via, file):
    """What went wrong with a publish to ``repo``, by the token's ``via``:
    Hugging Face refusing the token or the repository as both modes word it
    (hub_refusal), else as publish.ts's publishError says the rest."""
    if _unreachable(e):
        return f"Uploading {file}: {UNREACHABLE}" if file else UNREACHABLE
    refused = hub_refusal(_hub_status(e), repo, via)
    if refused:
        return refused
    reason = _hub_message(e)
    return f"Uploading {file} to {repo} failed: {reason}" if file else f"Publishing to {repo} failed: {reason}"


# --- Who is signed in -----------------------------------------------------------


def _who_is(token):
    """Who ``token`` is, ``{user, orgs, read_only}``, by Hugging Face's
    whoami-v2: the user's name, their organizations' (where they may publish
    too), and whether the token can only read. Raises huggingface_hub's
    error, or a ValueError for a token that is not a user's."""
    from huggingface_hub import whoami

    me = whoami(token=token)
    if me.get("type") != "user":
        raise ValueError(f"this is an {me.get('type')}'s token, not a user's: publish with your own")
    access = (me.get("auth") or {}).get("accessToken") or {}
    return {"user": me["name"], "orgs": [org["name"] for org in me.get("orgs") or []],
            "read_only": access.get("role") == "read"}


def _signed_in():
    """The token in use and who it is, ``{token, user, orgs, via, expires_at}``,
    or None: the one signed in here (a sign-in past its expiry is forgotten),
    else huggingface_hub's own, checked with whoami when it is first seen.
    A saved token that is refused is not kept: the next ask checks it again,
    since what failed may have been the network."""
    hf = get_session().huggingface
    signed_in = hf["signed_in"]
    if signed_in and signed_in["expires_at"] and signed_in["expires_at"] <= _now():
        logger.info(f"Hugging Face: the sign-in as {signed_in['user']} has expired")
        hf["signed_in"] = signed_in = None
    if signed_in:
        return signed_in
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        return None
    register_secret(token)
    if hf["saved"] is None or hf["saved"]["token"] != token:
        try:
            who = _who_is(token)
            refused = READ_ONLY if who["read_only"] else None
        except Exception as e:
            refused = _token_error(e)
        if refused:
            logger.warning(f"Hugging Face: the token saved on this server was not taken: {refused}")
            return None
        hf["saved"] = {"token": token, "user": who["user"], "orgs": who["orgs"], "via": "saved", "expires_at": None}
        logger.info(f"Hugging Face: signed in as {who['user']} with the token saved on this server")
    return hf["saved"]


def _account():
    """The credentials routes' answer: who is signed in and how, and the
    sign-in with a code that waits for the user (its code and where to enter
    it) or that failed (why). ``oauth_url`` is the browser's popup, which
    the server has none of; ``device_sign_in`` says Sign in is offered."""
    hf = get_session().huggingface
    who = _signed_in()
    device = hf["device"]
    if device is not None and device["status"] == "pending":
        # Never its device code, for which Hugging Face gives the token.
        device = {
            "status": "pending",
            "user_code": device["user_code"],
            "verification_uri": device["verification_uri"],
            "verification_uri_complete": device["verification_uri_complete"],
            "expires_at": _iso(device["expires_at"]),
        }
    return {
        "success": True,
        "signed_in": who is not None,
        "user": who["user"] if who else None,
        "orgs": who["orgs"] if who else [],
        "via": who["via"] if who else None,
        "expires_at": _iso(who["expires_at"]) if who and who["expires_at"] else None,
        "oauth_url": None,
        "device_sign_in": bool(HF_OAUTH_CLIENT_ID),
        "device": device,
        "can_remember": True,
    }


def _start_device_sign_in():
    """Ask Hugging Face for a sign-in code, and follow it in a thread
    (``_follow_device_sign_in``). It replaces any other waiting. Raises
    requests' error, or a RuntimeError with Hugging Face's reason for
    giving no code."""
    answer = requests.post(DEVICE_URL, data={
        "client_id": HF_OAUTH_CLIENT_ID, "scope": " ".join(HF_OAUTH_SCOPES),
    }, timeout=30)
    code = answer.json()
    if "device_code" not in code:
        raise RuntimeError(code.get("error_description") or code.get("error") or f"HTTP {answer.status_code}")
    register_secret(code["device_code"])
    device = {
        "status": "pending",
        "user_code": code["user_code"],
        "verification_uri": code["verification_uri"],
        "verification_uri_complete": code.get("verification_uri_complete"),
        "expires_at": _now() + timedelta(seconds=int(code.get("expires_in", 900))),
        "device_code": code["device_code"],
        "interval": int(code.get("interval", 5)),
    }
    hf = get_session().huggingface
    with hf["lock"]:
        hf["device"] = device
    logger.info(f"Hugging Face: waiting for the code {device['user_code']} to be entered at "
                f"{device['verification_uri']}")
    _start_thread(_follow_device_sign_in, hf, device)


def _follow_device_sign_in(hf, device):
    """Ask Hugging Face for ``device``'s token every interval (five seconds
    longer after each slow_down), until it gives it, refuses it, or the code
    expires; then sign in with it, or keep why not for the panel to say. A
    sign-in started again, a token pasted or a sign-out meanwhile ends it:
    the session's device is then another, or none."""
    interval = device["interval"]
    while True:
        time.sleep(interval)
        if hf["device"] is not device:
            return
        if _now() >= device["expires_at"]:
            error = EXPIRED
            break
        try:
            answer = requests.post(TOKEN_URL, data={
                "grant_type": DEVICE_GRANT, "device_code": device["device_code"], "client_id": HF_OAUTH_CLIENT_ID,
            }, timeout=30)
        except requests.RequestException:
            error = UNREACHABLE
            break
        try:
            body = answer.json()
        except ValueError:
            body = {}
        reason = body.get("error")
        if reason == "authorization_pending":
            continue
        if reason == "slow_down":
            interval += SLOW_DOWN_SECONDS
            continue
        if reason == "access_denied":
            error = ("Hugging Face did not sign you in: the request was denied at huggingface.co. "
                     "Sign in again to retry.")
        elif reason == "expired_token":
            error = EXPIRED
        elif reason or not body.get("access_token"):
            why = body.get("error_description") or reason or f"HTTP {answer.status_code}"
            error = f"Hugging Face did not sign you in: {why}"
        else:
            error = _device_signed_in(hf, device, body)
            if error is None:
                return
        break
    with hf["lock"]:
        if hf["device"] is device:
            hf["device"] = {"status": "failed", "error": error}
    logger.warning(error)


def _device_signed_in(hf, device, body):
    """Sign in with the token a device sign-in was given (``body``, the token
    endpoint's answer), unless it was ended meanwhile. None once signed in,
    else why not."""
    token = body["access_token"]
    register_secret(token)
    try:
        who = _who_is(token)
    except Exception as e:
        return _token_error(e)
    if who["read_only"]:
        return READ_ONLY
    expires_in = body.get("expires_in")
    with hf["lock"]:
        if hf["device"] is not device:
            return None
        hf["device"] = None
        hf["signed_in"] = {"token": token, "user": who["user"], "orgs": who["orgs"], "via": "oauth",
                           "expires_at": _now() + timedelta(seconds=int(expires_in)) if expires_in else None}
    logger.info(f"Hugging Face: signed in as {who['user']} (kept in this dashboard's memory only)")
    return None


@finetune_bp.route("/api/huggingface/credentials", methods=["GET"])
def huggingface_account():
    return jsonify(_account())


@finetune_bp.route("/api/huggingface/credentials", methods=["POST"])
def huggingface_sign_in():
    """A pasted token ({token, remember?}), checked with whoami and kept in
    place of any (refused when it can only read); Sign in with Hugging Face
    ({device: true}); or signing out ({forget: true}), which forgets the token
    signed in here and leaves a saved one. Any of them ends a sign-in that
    waits for its code."""
    body, refused = parse(HuggingFaceCredentials, request.get_json(silent=True))
    if refused:
        return refused
    hf = get_session().huggingface
    with hf["lock"]:
        hf["device"] = None
    if body.forget:
        if hf["signed_in"]:
            logger.info(f"Hugging Face: signed out ({hf['signed_in']['user']}'s token forgotten)")
        hf["signed_in"] = None
        return jsonify(_account())
    if body.device:
        if not HF_OAUTH_CLIENT_ID:
            return _error("This cellmap-flow has no Hugging Face app to sign in with: paste an access token.", 400)
        try:
            _start_device_sign_in()
        except Exception as e:
            message = UNREACHABLE if _unreachable(e) else f"Hugging Face would not start a sign-in: {_hub_message(e)}"
            logger.warning(message)
            return _error(message, 502)
        return jsonify(_account())
    token = (body.token or "").strip()
    if not token.startswith("hf_"):
        return _error("Paste a Hugging Face access token: it starts with hf_ "
                      "(make one at https://huggingface.co/settings/tokens).", 400)
    register_secret(token)
    try:
        who = _who_is(token)
    except Exception as e:
        return _error(_token_error(e), 401)
    if who["read_only"]:
        return _error(READ_ONLY, 403)
    if body.remember:
        from huggingface_hub import login

        try:
            login(token=token, add_to_git_credential=False)
        except Exception as e:
            return _error(f"The token is good, but could not be saved on this server: {_hub_message(e)}", 500)
    hf["signed_in"] = {"token": token, "user": who["user"], "orgs": who["orgs"],
                       "via": "saved" if body.remember else "token", "expires_at": None}
    where = ("saved on this server, as `hf auth login` saves it" if body.remember
             else "kept in this dashboard's memory only")
    logger.info(f"Hugging Face: signed in as {who['user']} with an access token ({where})")
    return jsonify(_account())


# --- Publishing -------------------------------------------------------------------


def _served_job(name):
    """The finetune job whose served model is ``name`` (its latest
    iteration's) and that wrote a serving YAML for it, or None."""
    jobs = list(get_session().finetune_job_manager.jobs.values())
    return next((job for job in reversed(jobs)
                 if job.finetuned_model_name == name and job.model_yaml_path), None)


def _answer(publish):
    """A publish as the routes answer it (publish.ts's answerOf). Done, it
    says where the model is, with the links on the public app (the
    dashboard is not the app) and the dataset open now."""
    with get_session().huggingface["lock"]:
        publish = dict(publish)
    answer = {"success": True, "status": publish["status"], "repo": publish["repo"], "folder": publish["folder"]}
    if publish["status"] == "running":
        answer.update(step=publish["step"], file=publish["file"], fraction=publish["fraction"])
    elif publish["status"] == "failed":
        answer["error"] = publish["error"]
    else:
        answer.update(published_links(publish["repo"], publish["folder"], PUBLIC_APP,
                                       _app_dataset(get_session().dataset_path)))
        answer.update(at=publish["at"], created=publish["created"], private=publish["private"])
    return answer


def _app_dataset(dataset):
    """``dataset`` as the app could open it, in a browser anywhere: a URL
    (s3://, https://, a zarr:// one...), not a path on this server's disks,
    which a link or a model card would name to no use; None for a path."""
    dataset = str(dataset or "")
    return dataset if re.match(r"[a-z][a-z0-9+.-]*://", dataset) and not dataset.startswith("file:") else None


@finetune_bp.route("/api/finetune/model/<name>/publish", methods=["POST"])
def start_publish(name):
    """The model ``name`` published to ``repo`` (``owner/name``; blank: the
    user's own, named after the model). What can be refused at once is: no
    such model, no token, a repository name Hugging Face would refuse, an
    owner that is neither the user nor one of their organizations, a publish
    of it under way."""
    sent = request.get_json(silent=True)
    body, refused = parse(FinetunePublish, {} if sent is None else sent)
    if refused:
        return refused
    job = _served_job(name)
    if job is None:
        return _error(f"No model finetuned on this server is named {name}", 404)
    who = _signed_in()
    if who is None:
        return _error(SIGN_IN_FIRST, 401)
    repo = (body.repo or "").strip() or f"{who['user']}/{name}"
    refused = repo_refusal(repo, who["user"], who["orgs"])
    if refused:
        return _error(refused["error"], refused["status"])
    hf = get_session().huggingface
    with hf["lock"]:
        if hf["publishes"].get(name, {}).get("status") == "running":
            return _error(f"{name} is being published already: wait for it to finish.", 409)
        publish = {"repo": repo, "folder": name, "status": "running", "step": "exporting", "file": None,
                   "fraction": None}
        hf["publishes"][name] = publish
    _start_thread(_run, hf, publish, job, who, body.private, body.include_dataset)
    return jsonify(_answer(publish))


@finetune_bp.route("/api/finetune/model/<name>/publish", methods=["GET"])
def publish_status(name):
    """This dashboard's publish of the model ``name`` (running, done or failed), else status "none"."""
    if _served_job(name) is None:
        return _error(f"No model finetuned on this server is named {name}", 404)
    publish = get_session().huggingface["publishes"].get(name)
    return jsonify(_answer(publish) if publish else {"success": True, "status": "none"})


def _run(hf, publish, job, who, private, include_dataset):
    """The publish itself: the export, the files (portable.publish's card
    and manifest, then the export's files), the commit, and done; or failed,
    with why."""
    def update(**fields):
        with hf["lock"]:
            publish.update(fields)

    try:
        served = yaml.safe_load(Path(job.model_yaml_path).read_text())
        folder, manifest, export_dir = _export(job)
        published_at = _iso(_now())
        pub = {
            "repo": publish["repo"],
            "folder": folder,
            "published_at": published_at,
            "manifest": manifest,
            "origin": "trained",
            "where": "server",
            **_training(job, served["models"][0]),
            "app_version": cellmap_flow.__version__,
            "dataset": _app_dataset(served.get("data_path")) if include_dataset else None,
        }
        files = [(f["path"], f["text"].encode()) for f in publication_files(pub)]
        files += [(f"{folder}/{v['file']}", export_dir / v["file"]) for v in manifest["variants"]]
        update(folder=folder)
        created, is_private = _commit(who["token"], publish["repo"], private, files,
                                      f"Publish {folder} from cellmap-flow", update)
        update(status="done", created=created, private=is_private, at=published_at)
        logger.info(f"Published {folder} to {HUB}/{publish['repo']} as {ref_of(publish['repo'], folder)}"
                    f"{' (a new repository)' if created else ''}")
    except Exception as e:
        error = _publish_error(e, publish["repo"], who["via"], publish["file"])
        update(status="failed", error=error)
        # The traceback for what is not Hugging Face's answer: an export or a
        # file that is not what it should be.
        logger.error(error, exc_info=_hub_status(e) is None and not _unreachable(e))


def _export(job):
    """The model exported for the web by finetune.export_web's cluster job,
    waited for: ``(its folder, its manifest, the folder's directory)``.
    Raises RuntimeError with the export's error."""
    from cellmap_flow.finetune import export_web

    export = export_web.start_export(job.model_yaml_path, job.corrections_path, job.output_dir)
    while (state := export_web.export_status(export))["status"] == "running":
        time.sleep(EXPORT_POLL_SECONDS)
    if state["status"] != "done":
        raise RuntimeError(f"its export to ONNX failed: {state['error']}")
    return state["folder"], state["manifest"], Path(state["dir"]) / state["folder"]


def _training(job, entry):
    """What the job and its serving YAML's model ``entry`` say of the
    model's training, as ``pub`` takes it. The epochs and the loss are the
    job's latest, which are the served iteration's while it is served,
    the only time the panel offers Publish. The YAML is written as its
    iteration finishes."""
    trained = datetime.fromtimestamp(Path(job.model_yaml_path).stat().st_mtime, timezone.utc)
    return {
        "finetuned_from": _finetuned_from(entry.get("base_model") or {}),
        "finetune_mode": "lora" if entry.get("lora_adapter_path") else "full",
        "epochs": job.current_epoch or None,
        "final_loss": job.latest_loss,
        "trained_at": _iso(trained),
    }


def _finetuned_from(base):
    """The ref of the model ``base`` (a serving YAML's ``base_model``) names,
    as *Add a model* takes it: a Hugging Face model's repo; for a model
    finetuned here before, where this dashboard published it, else what that
    one was finetuned from; None for a model known only by its files on this
    server (a checkpoint, a script)."""
    publishes = get_session().huggingface["publishes"]
    while base.get("type") == "finetune":
        published = publishes.get(base.get("name")) or {}
        if published.get("status") == "done":
            return ref_of(published["repo"], published["folder"])
        base = base.get("base_model") or {}
    return base.get("repo") if base.get("type") == "huggingface" else None


def _commit(token, repo, private, files, title, update):
    """Make ``repo`` (a model repository; private when asked) unless it is
    there already, and upload ``files`` (``(path in the repository, bytes or
    a local path)``) to it in one commit titled ``title``, telling
    ``update`` the step and the file. A repository that is there already
    takes the files as a new commit, its visibility as it was. Then the
    repository is read back, so that done means Hugging Face shows each file
    at the size sent (else this raises). Returns whether the repository was
    made, and whether Hugging Face says it is private, whatever was asked."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    update(step="creating", file=None, fraction=None)
    created = not api.repo_exists(repo)
    api.create_repo(repo, private=private, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=content) for path, content in files]
    # One file at a time, so the panel can say which; the commit then sends
    # none of them again.
    for operation in operations:
        update(step="uploading", file=operation.path_in_repo)
        api.preupload_lfs_files(repo, additions=[operation])
    update(step="committing", file=None)
    api.create_commit(repo, operations=operations, commit_message=title)
    update(step="checking")
    shown = {entry.path: entry.lfs.size if entry.lfs else entry.size
             for entry in api.get_paths_info(repo, [operation.path_in_repo for operation in operations])}
    absent = [op.path_in_repo for op in operations if shown.get(op.path_in_repo) != op.upload_info.size]
    if absent:
        raise RuntimeError(f"the commit went through, but Hugging Face does not show {', '.join(absent)} "
                           "at the size uploaded")
    return created, api.model_info(repo).private
