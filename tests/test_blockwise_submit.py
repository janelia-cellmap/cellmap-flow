from types import SimpleNamespace

import pytest
from flask import Flask

from cellmap_flow.dashboard.routes import blockwise as blockwise_routes


TASK_NAME = "nuc_cerebellum_20260903_120000"


def _submit(monkeypatch, tmp_path, job_name="nuc cerebellum", gen_task_name=TASK_NAME):
    app = Flask(__name__)
    app.register_blueprint(blockwise_routes.blockwise_bp)
    # daisy_logs/ is created relative to the cwd; keep it out of the repo.
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        blockwise_routes,
        "validate_blockwise",
        lambda: {"valid": True},
    )
    gen = {"success": True, "task_paths": ["/tmp/task.yaml"]}
    if gen_task_name:
        gen["task_name"] = gen_task_name
    monkeypatch.setattr(blockwise_routes, "generate_blockwise_task", lambda: gen)

    captured = {}

    def fake_run(cmd, capture_output, text, env):
        captured["cmd"] = cmd
        return SimpleNamespace(
            returncode=0,
            stdout="Job <12345> is submitted to default queue.",
            stderr="",
        )

    monkeypatch.setattr(blockwise_routes.subprocess, "run", fake_run)

    response = app.test_client().post(
        "/api/blockwise/submit",
        json={
            "job_name": job_name,
            "pipeline": {
                "blockwise_config": [
                    {
                        "params": {
                            "nb_cores_master": 4,
                            "charge_group": "cellmap",
                            "queue": "gpu_h100",
                        }
                    }
                ]
            },
        },
    )
    assert response.status_code == 200
    return response.get_json(), captured["cmd"]


def test_submit_blockwise_master_stays_cpu_only(monkeypatch, tmp_path):
    body, cmd = _submit(monkeypatch, tmp_path)
    assert body["success"] is True
    assert "-gpu" not in cmd
    assert "-q" not in cmd
    assert cmd[-4:] == [
        "python",
        "-m",
        "cellmap_flow.blockwise.multiple_cli",
        "/tmp/task.yaml",
    ]


def test_submit_blockwise_master_named_after_task(monkeypatch, tmp_path):
    body, cmd = _submit(monkeypatch, tmp_path)
    # The master's -J is the generated task name, not an unrelated timestamp.
    assert cmd[cmd.index("-J") + 1] == TASK_NAME
    assert body["job_name"] == TASK_NAME
    assert body["task_name"] == TASK_NAME


def test_submit_blockwise_master_has_per_job_log(monkeypatch, tmp_path):
    body, cmd = _submit(monkeypatch, tmp_path)
    expected = f"daisy_logs/{TASK_NAME}.master.%J.log"
    assert cmd[cmd.index("-o") + 1] == expected
    assert cmd[cmd.index("-e") + 1] == expected
    assert body["log_file"] == f"daisy_logs/{TASK_NAME}.master.12345.log"
    assert (tmp_path / "daisy_logs").is_dir()


def test_submit_falls_back_to_job_name_when_generate_has_no_task_name(monkeypatch, tmp_path):
    body, cmd = _submit(monkeypatch, tmp_path, job_name="my run", gen_task_name=None)
    j = cmd[cmd.index("-J") + 1]
    assert j.startswith("my_run_")
    assert body["task_name"] == j


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("nuc cerebellum", "nuc_cerebellum"),
        ("  a/b\\c:d  ", "a_b_c_d"),
        ("ok-name.v2", "ok-name.v2"),
        ("___", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_sanitize_job_name(raw, expected):
    assert blockwise_routes._sanitize_job_name(raw) == expected


def test_make_task_name_defaults_when_empty():
    assert blockwise_routes._make_task_name("", "20260903_120000") == "cellmap_flow_20260903_120000"
    assert blockwise_routes._make_task_name("nuc test", "20260903_120000") == "nuc_test_20260903_120000"


def test_spawn_worker_uses_per_task_per_job_logs(monkeypatch, tmp_path):
    from cellmap_flow.blockwise import blockwise_processor as bp

    monkeypatch.chdir(tmp_path)
    captured = {}
    monkeypatch.setattr(bp.subprocess, "run", lambda cmd, **kw: captured.setdefault("cmd", cmd))

    name = f"predict_setup55_{TASK_NAME}"
    bp.spawn_worker(name, "/tmp/task.yaml", "cellmap", "gpu_h100", ncpu=4)()

    cmd = captured["cmd"]
    assert cmd[cmd.index("-J") + 1] == name
    assert cmd[cmd.index("-o") + 1] == f"daisy_logs/{name}/lsf_worker.%J.out"
    assert cmd[cmd.index("-e") + 1] == f"daisy_logs/{name}/lsf_worker.%J.err"
    assert (tmp_path / "daisy_logs" / name).is_dir()
    assert cmd[-3:] == ["cellmap_flow_blockwise", "/tmp/task.yaml", "--client"]
