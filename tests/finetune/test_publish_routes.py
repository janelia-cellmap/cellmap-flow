"""Publishing a model finetuned on the server to Hugging Face (routes/finetune/publish.py).

Hugging Face is huggingface_hub's functions, replaced by ``FakeHub``: who a
token is, the token ``hf auth login`` saved, and the repositories with the
files committed to them. The device sign-in's two endpoints are requests'
POSTs, answered in turn. The export (finetune.export_web) is replaced by one
that writes its files at once: the server form's, and a browser form's
manifest and variants unless it is told there is none. The routes' threads run in
the request that starts them, but a device sign-in's, which a test runs when
it has seen the sign-in wait.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cellmap_flow.ai_annotate import secrets
from cellmap_flow.dashboard.routes.finetune import publish as routes
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune import export_web

TOKEN = "hf_pasted0123456789abcdefghijklmnopqrst"
SAVED = "hf_saved00123456789abcdefghijklmnopqrst"
MODEL = "mito_finetuned_20261026_120000"
CREDENTIALS = "/api/huggingface/credentials"
PUBLISH = f"/api/finetune/model/{MODEL}/publish"


class _Refused(Exception):
    """An error answer of the Hub's, as huggingface_hub raises it: with its response's status."""

    def __init__(self, status):
        super().__init__(f"{status} Client Error for url: https://huggingface.co/api/x")
        self.response = SimpleNamespace(status_code=status)


class FakeHub:
    """Hugging Face as huggingface_hub's functions see it: ``users`` by token
    (``{user, orgs, role}``), the token saved on this machine (``saved``),
    and ``repos`` by name, each ``{private, files: {path: bytes}}``;
    ``from_disk``, the files the last commit was given as paths to read
    rather than as bytes. ``refuse_commit`` makes a commit fail with that
    status."""

    def __init__(self):
        self.users = {}
        self.saved = None
        self.logins = []
        self.repos = {}
        self.from_disk = set()
        self.refuse_commit = None
        hub = self

        class Api:
            def __init__(self, token=None):
                self.token = token

            def repo_exists(self, repo):
                return repo in hub.repos

            def create_repo(self, repo, private, exist_ok):
                hub.repos.setdefault(repo, {"private": private, "files": {}})

            def preupload_lfs_files(self, repo, additions):
                pass

            def create_commit(self, repo, operations, commit_message):
                assert self.token in hub.users, "committed with a token Hugging Face does not know"
                if hub.refuse_commit:
                    raise _Refused(hub.refuse_commit)
                hub.from_disk = {op.path_in_repo for op in operations if not isinstance(op.path_or_fileobj, bytes)}
                for op in operations:
                    content = op.path_or_fileobj
                    files = hub.repos[repo]["files"]
                    files[op.path_in_repo] = content if isinstance(content, bytes) else Path(content).read_bytes()

            def get_paths_info(self, repo, paths):
                files = hub.repos[repo]["files"]
                return [SimpleNamespace(path=p, size=len(files[p]), lfs=None) for p in paths if p in files]

            def model_info(self, repo):
                return SimpleNamespace(private=hub.repos[repo]["private"])

        self.Api = Api

    def whoami(self, token):
        if token not in self.users:
            raise _Refused(401)
        who = self.users[token]
        return {"type": "user", "name": who["user"], "orgs": [{"name": o} for o in who.get("orgs", [])],
                "auth": {"accessToken": {"role": who.get("role", "write")}}}

    def login(self, token, add_to_git_credential):
        self.logins.append(add_to_git_credential)
        self.saved = token


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub

    fake = FakeHub()
    monkeypatch.setattr(huggingface_hub, "whoami", fake.whoami)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: fake.saved)
    monkeypatch.setattr(huggingface_hub, "login", fake.login)
    monkeypatch.setattr(huggingface_hub, "HfApi", fake.Api)
    monkeypatch.setattr(routes, "_start_thread", lambda target, *args: target(*args))
    yield fake
    secrets._clear_registered_secrets()


@pytest.fixture
def served(make_job, tmp_path, monkeypatch):
    """A job serving MODEL, a LoRA finetune of a Hugging Face model, and an
    export that gives its server form and a browser form of two variants,
    or, with ``no_browser`` set to why, the server form alone. Returns
    ``{started: the exports started, no_browser}``."""
    yaml_path = tmp_path / "models" / f"{MODEL}.yaml"
    yaml_path.parent.mkdir()
    yaml_path.write_text(yaml.safe_dump({"data_path": "s3://bucket/raw.zarr", "models": [{
        "type": "finetune", "name": MODEL, "lora_adapter_path": "/runs/r/lora_adapter",
        "base_model": {"type": "huggingface", "repo": "cellmap/mito-base"},
    }]}))
    job = make_job(finetuned_model_name=MODEL, model_yaml_path=yaml_path, current_epoch=3, latest_loss=0.25,
                   inference_server_ready=True, network_parameters=2_000_000)
    get_session().finetune_job_manager = SimpleNamespace(jobs={job.job_id: job})

    export = SimpleNamespace(started=[], no_browser=None)

    variants = [{"variant": v, "file": f"model_{v}.onnx", "input_size": [64] * 3, "output_size": [32] * 3}
                for v in ("fp32", "fp16")]
    manifest = {"task": "mito", "channels": ["mito"], "input_voxel_size_nm": [8] * 3,
                "output_voxel_size_nm": [8] * 3}
    server = {"files": ["metadata.json", "model.ts", "model.pt2", "export_record.json"], "input_size": [64] * 3,
              "output_size": [32] * 3}

    # The export job, done at once: its files written where start_export says.
    def start_export(yaml_path, corrections_dir, run_dir):
        export.started.append(yaml_path)
        out = Path(run_dir) / "web_export" / Path(yaml_path).stem
        out.mkdir(parents=True)
        for name in server["files"]:
            (out / name).write_bytes(name.encode() * 100)
        if not export.no_browser:
            (out / MODEL).mkdir()
            for v in variants:
                (out / MODEL / v["file"]).write_bytes(v["variant"].encode() * 100)
        return out

    def export_status(out):
        return {"status": "done", "error": None, "dir": str(out), "folder": MODEL,
                "manifest": {**manifest, "variants": [] if export.no_browser else variants},
                "server": server, "no_browser": export.no_browser}

    monkeypatch.setattr(export_web, "start_export", start_export)
    monkeypatch.setattr(export_web, "export_status", export_status)
    return export


def test_a_pasted_token_publishes_the_served_model_in_both_forms_with_its_card_and_manifest(dashboard, hub, served,
                                                                                            caplog):
    hub.users[TOKEN] = {"user": "ada", "orgs": ["lab"]}
    caplog.set_level(logging.INFO, logger="cellmap_flow")

    assert dashboard.get(CREDENTIALS).get_json()["signed_in"] is False
    account = dashboard.post(CREDENTIALS, json={"token": TOKEN}).get_json()
    assert (account["signed_in"], account["user"], account["via"]) == (True, "ada", "token")
    assert secrets.redact(TOKEN) == secrets.REDACTED

    # The panel says what model.pt2 adds for this model: its 2M parameters in float32.
    assert dashboard.get(PUBLISH).get_json() == {"success": True, "status": "none", "lora_copy_bytes": 8_000_000}
    started = dashboard.post(PUBLISH, json={"repo": "lab/mito", "include_dataset": True})
    assert started.status_code == 200, started.get_json()
    done = dashboard.get(PUBLISH).get_json()

    assert done["status"] == "done", done
    assert (done["ref"], done["created"], done["private"]) == (f"lab/mito/{MODEL}", True, False)
    assert (done["server_ref"], done["no_browser"]) == ("lab/mito", None)
    assert done["try_link"].endswith(f"?model=lab/mito/{MODEL}")
    files = hub.repos["lab/mito"]["files"]
    model_files = ["metadata.json", "model.ts", "model.pt2", "export_record.json", f"{MODEL}/model_fp32.onnx",
                   f"{MODEL}/model_fp16.onnx"]
    assert sorted(files) == sorted(["README.md", f"{MODEL}/manifest.json", *model_files])
    assert hub.from_disk == set(model_files), "the model's files are read from the disk as they are sent"
    published = json.loads(files[f"{MODEL}/manifest.json"])["published"]
    assert published["finetuned_from"] == "cellmap/mito-base"
    assert (published["finetune_mode"], published["epochs"], published["final_loss"]) == ("lora", 3, 0.25)
    assert published["dataset"] == "s3://bucket/raw.zarr"
    # Nothing answered or logged carries the token.
    assert TOKEN not in json.dumps([account, done]) + caplog.text


def test_what_is_refused_at_once(dashboard, hub, served):
    assert dashboard.post(PUBLISH, json={}).status_code == 401, "no token yet"
    hub.users[TOKEN] = {"user": "ada"}
    hub.users["hf_readonly0123456789abcdefghijklmn"] = {"user": "ada", "role": "read"}
    assert dashboard.post(CREDENTIALS, json={"token": "hf_unknown0123456789abcdefghijklmn"}).status_code == 401
    assert dashboard.post(CREDENTIALS, json={"token": "hf_readonly0123456789abcdefghijklmn"}).status_code == 403
    assert dashboard.post(CREDENTIALS, json={"token": "not-a-token"}).status_code == 400
    dashboard.post(CREDENTIALS, json={"token": TOKEN})

    assert dashboard.post("/api/finetune/model/nothing/publish", json={}).status_code == 404
    assert dashboard.post(PUBLISH, json={"repo": "ada/-bad-"}).status_code == 400
    assert dashboard.post(PUBLISH, json={"repo": "someone-else/mito"}).status_code == 403
    assert served.started == [] and hub.repos == {}, "nothing was exported or made"


def test_a_model_with_no_browser_form_publishes_its_server_form_alone(dashboard, hub, served):
    served.no_browser = "The network has 812,345,678 parameters, 3.2 GB in float32: over the 2 GB an .onnx file holds"
    hub.users[TOKEN] = {"user": "ada"}
    dashboard.post(CREDENTIALS, json={"token": TOKEN})

    dashboard.post(PUBLISH, json={"include_dataset": True, "lora_copy": False})
    done = dashboard.get(PUBLISH).get_json()

    assert done["status"] == "done", done
    assert (done["server_ref"], done["no_browser"]) == (f"ada/{MODEL}", served.no_browser)
    assert (done["ref"], done["try_link"], done["dataset_link"]) == (None, None, None), "nothing opens it in the browser"
    files = hub.repos[f"ada/{MODEL}"]["files"]
    assert sorted(files) == ["README.md", "export_record.json", "metadata.json", "model.ts"], "no model.pt2, as asked"
    card = files["README.md"].decode()
    assert f"`ada/{MODEL}`" in card and served.no_browser in card
    assert "?model=" not in card, "no link to the app, which could not open it"


def test_a_commit_the_hub_refuses_fails_the_publish(dashboard, hub, served):
    hub.users[TOKEN] = {"user": "ada"}
    dashboard.post(CREDENTIALS, json={"token": TOKEN})
    hub.refuse_commit = 500

    dashboard.post(PUBLISH, json={})
    answer = dashboard.get(PUBLISH).get_json()

    assert answer["status"] == "failed" and f"ada/{MODEL}" in answer["error"]
    assert "https://" not in answer["error"], "huggingface_hub's URLs are left out"


def test_a_saved_token_is_used_and_a_remembered_one_is_saved(dashboard, hub, served):
    hub.users[SAVED] = {"user": "ada"}
    hub.saved = SAVED
    account = dashboard.get(CREDENTIALS).get_json()
    assert (account["signed_in"], account["via"], account["can_remember"]) == (True, "saved", True)

    hub.users[TOKEN] = {"user": "bob"}
    account = dashboard.post(CREDENTIALS, json={"token": TOKEN, "remember": True}).get_json()
    assert (account["user"], account["via"]) == ("bob", "saved")
    assert (hub.saved, hub.logins) == (TOKEN, [False])
    # Signing out forgets the token in memory; the saved one is still there.
    assert dashboard.post(CREDENTIALS, json={"forget": True}).get_json()["user"] == "bob"


def _device_endpoints(monkeypatch, token_answers):
    """Hugging Face's device and token endpoints, the token endpoint giving
    ``token_answers`` in turn; returns the asks it got."""
    asked = []

    def post(url, data, timeout):
        asked.append((url, dict(data)))
        if url == routes.DEVICE_URL:
            body = {"device_code": "device-secret-0123456789", "user_code": "ABCD-EFGH",
                    "verification_uri": "https://huggingface.co/device", "expires_in": 900, "interval": 5}
        else:
            body = token_answers.pop(0)
        return SimpleNamespace(json=lambda: body, status_code=400 if "error" in body else 200)

    monkeypatch.setattr(routes.requests, "post", post)
    monkeypatch.setattr(routes.time, "sleep", lambda seconds: asked.append(("slept", seconds)))
    monkeypatch.setattr(routes, "HF_OAUTH_CLIENT_ID", "https://app.example/.well-known/oauth-cimd")
    return asked


def test_signing_in_with_a_code_waits_for_it_then_signs_in(dashboard, hub, monkeypatch):
    hub.users["hf_oauth0123456789abcdefghijklmnopq"] = {"user": "ada"}
    asked = _device_endpoints(monkeypatch, [
        {"error": "authorization_pending"},
        {"error": "slow_down"},
        {"access_token": "hf_oauth0123456789abcdefghijklmnopq", "expires_in": 3600},
    ])
    follow = []
    monkeypatch.setattr(routes, "_start_thread", lambda target, *args: follow.append((target, args)))

    waiting = dashboard.post(CREDENTIALS, json={"device": True}).get_json()
    assert waiting["signed_in"] is False and waiting["device_sign_in"] is True
    assert (waiting["device"]["status"], waiting["device"]["user_code"]) == ("pending", "ABCD-EFGH")
    assert "device-secret" not in json.dumps(waiting)

    [(target, args)] = follow
    target(*args)
    account = dashboard.get(CREDENTIALS).get_json()

    assert (account["signed_in"], account["user"], account["via"]) == (True, "ada", "oauth")
    assert account["expires_at"] and account["device"] is None
    assert [s for _, s in asked if _ == "slept"] == [5, 5, 10], "slow_down asks again five seconds later"
    assert hub.saved is None, "a sign-in's token is not saved"


@pytest.mark.parametrize("answer", [{"error": "access_denied"}, {"error": "expired_token"}])
def test_a_code_denied_or_expired_says_so(dashboard, hub, monkeypatch, answer):
    _device_endpoints(monkeypatch, [answer])

    dashboard.post(CREDENTIALS, json={"device": True})
    account = dashboard.get(CREDENTIALS).get_json()

    assert account["signed_in"] is False
    assert account["device"]["status"] == "failed" and account["device"]["error"]
