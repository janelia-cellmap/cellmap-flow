"""What the dashboard shows while training runs, and what a restart picks up.

Three separate ways a run looked wrong without being wrong, or looked fine
without being fine:

  - the loss on screen arrived five to ten epochs at a time, because tee
    block-buffers its writes to a file even when the writer flushes;
  - raising the LoRA rank on restart silently shrank every update, because
    peft scales by lora_alpha / r and only r was being changed;
  - "continue with my new annotations" trained on the old ones, because the
    restart skipped the sync that puts them on disk.
"""

import argparse

import pytest


class _Job:
    """Just the progress fields _parse_training_progress writes."""

    def __init__(self):
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_batch = 0
        self.total_batches = 0
        self.latest_loss = None


@pytest.fixture
def manager():
    from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager

    return FinetuneJobManager.__new__(FinetuneJobManager)


class TestEpochLosses:
    """One point per epoch, from the epoch summary line.

    Per-batch losses are deliberately not read. What made the display look
    stuck was tee buffering the log file, not the reporting interval -- see
    TestTheLogReachesDiskAsItIsWritten.
    """

    def test_an_epoch_summary_sets_the_loss(self, manager):
        job = _Job()
        manager._parse_training_progress(job, """
Starting epoch 1 of 100...
  Batch 1/3 - Loss: 0.233847 (sup: 0.233847, distill: 0.000000)
Epoch 1/100 - Loss: 0.218966 - Supervised: 0.218809
""")
        assert job.latest_loss == pytest.approx(0.218966)
        assert (job.current_epoch, job.total_epochs) == (1, 100)

    def test_batch_losses_are_ignored(self, manager):
        """A batch loss is a running mean mid-epoch, not an epoch's result."""
        job = _Job()
        manager._parse_training_progress(job, """
Starting epoch 4 of 10...
  Batch 1/3 - Loss: 0.9
  Batch 2/3 - Loss: 0.8
""")
        assert job.latest_loss is None
        assert job.current_epoch == 4, "the counter should still advance"

    def test_the_last_epoch_in_the_chunk_wins(self, manager):
        job = _Job()
        manager._parse_training_progress(job, """
Epoch 1/10 - Loss: 0.9 - Supervised: 0.9
Starting epoch 2 of 10...
Epoch 2/10 - Loss: 0.4 - Supervised: 0.4
""")
        assert job.current_epoch == 2
        assert job.latest_loss == pytest.approx(0.4)

    def test_a_chunk_of_only_batches_keeps_the_epoch_it_had(self, manager):
        """The log is read incrementally, so a chunk can be batches only."""
        job = _Job()
        job.current_epoch, job.total_epochs = 7, 10
        job.latest_loss = 0.5
        manager._parse_training_progress(job, "  Batch 2/3 - Loss: 0.123\n")
        assert (job.current_epoch, job.latest_loss) == (7, 0.5)


class TestTheLogReachesDiskAsItIsWritten:
    def test_tee_is_line_buffered(self, tmp_path):
        """The trainer flushes every line it prints, but tee writes to the log
        file through stdio, which block-buffers to a file -- so roughly 8KB,
        five to ten epochs' worth, landed at once and the dashboard showed
        nothing in between."""
        import inspect

        from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager

        src = inspect.getsource(FinetuneJobManager)
        assert "| stdbuf -oL tee " in src, "tee must be line-buffered too"


class TestRankChangeKeepsItsScaling:
    """peft scales the adapter by lora_alpha / r.

    Submit derives alpha = 2 * r. A restart carried only lora_r, so going
    from r=8 to r=64 took the scaling from 16/8 = 2.0 to 16/64 = 0.25: every
    update an eighth of its former size, and a loss curve that looked smooth
    because almost nothing was moving.
    """

    def _args(self, **kw):
        base = dict(lora_r=8, lora_alpha=16, output_dir=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_raising_the_rank_raises_alpha_with_it(self):
        from cellmap_flow.finetune.finetune_cli import _apply_restart_params

        args = self._args()
        _apply_restart_params(args, {"params": {"lora_r": 64}})
        assert args.lora_r == 64
        assert args.lora_alpha == 128
        assert args.lora_alpha / args.lora_r == 2.0

    def test_an_explicit_alpha_is_left_alone(self):
        from cellmap_flow.finetune.finetune_cli import _apply_restart_params

        args = self._args()
        _apply_restart_params(args, {"params": {"lora_r": 64, "lora_alpha": 16}})
        assert args.lora_alpha == 16

    def test_a_restart_that_does_not_touch_the_rank_changes_nothing(self):
        from cellmap_flow.finetune.finetune_cli import _apply_restart_params

        args = self._args()
        _apply_restart_params(args, {"params": {"num_epochs": 50}})
        assert (args.lora_r, args.lora_alpha) == (8, 16)


class TestRestartPicksUpNewAnnotations:
    def test_the_manifest_path_no_longer_skips_the_sync(self):
        """A manifest used to mean "skip the sync entirely".

        The trainer reads the volume zarr on disk, and only the sync puts the
        browser's strokes there -- the background thread runs every 30s, so
        annotating and immediately continuing trained on stale data.
        """
        import inspect

        from cellmap_flow.dashboard.routes.finetune import training

        src = inspect.getsource(training.restart_finetuning_job_response)
        body = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        assert "sync_all_annotations_from_minio" in body
        assert "skipping pre-restart" not in body


class TestFinetunedLayerDisplayRange:
    def test_a_sigmoid_output_is_shown_over_0_to_1(self, manager, monkeypatch):
        """It was hardcoded to range=[0, 255], so a 0-1 output rendered black
        -- the finetuned layer looked far worse than the same model added
        through the normal path."""
        from cellmap_flow.utils import server_info

        monkeypatch.setattr(
            server_info, "fetch_model_info", lambda *a, **k: {"output_class": None}
        )
        from cellmap_flow.globals import g

        class _Sigmoid:
            def to_dict(self):
                return {"name": "SigmoidPostprocessor"}

        monkeypatch.setattr(g, "postprocess", [_Sigmoid()], raising=False)

        shader = manager._finetuned_shader("http://example:8000")
        assert "range=[0, 1]" in shader
        assert "255" not in shader


class TestRestartSyncIsReported:
    """Asking MinIO whether anything changed is the check, not the cost.

    The browser writes strokes straight to MinIO, so there is no local signal
    to consult -- force=False diffs chunk keys and downloads only what
    differs. A parameters-only restart therefore pulls nothing, and the log
    and the response both say which of the two happened.
    """

    def _call(self, monkeypatch, pulled):
        from cellmap_flow.dashboard.app import app
        from cellmap_flow.dashboard.routes.finetune import training

        monkeypatch.setattr(
            training, "sync_all_annotations_from_minio", lambda force=True: pulled
        )
        monkeypatch.setattr(training, "build_restart_params", lambda data: {})

        class _Manager:
            jobs = {}

            def restart_finetuning_job(self, job_id, updated_params):
                return type("J", (), {"job_id": job_id})()

        monkeypatch.setattr(training.g, "finetune_job_manager", _Manager(),
                            raising=False)
        with app.test_request_context():
            response = training.restart_finetuning_job_response("job-1", {})
        return response.get_json()

    def test_new_annotations_are_announced(self, monkeypatch):
        body = self._call(monkeypatch, 2)
        assert body["annotations_synced"] == 2
        assert "2 volume(s)" in body["message"]

    def test_a_parameters_only_restart_says_it_pulled_nothing(self, monkeypatch):
        body = self._call(monkeypatch, 0)
        assert body["annotations_synced"] == 0
        assert "No new annotations" in body["message"]

    def test_minio_being_down_is_not_reported_as_a_pull(self, monkeypatch):
        """sync_all_annotations_from_minio returns -1 when MinIO is absent."""
        body = self._call(monkeypatch, -1)
        assert body["annotations_synced"] == 0
