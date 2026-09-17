"""What the dashboard shows while training runs, and what a restart picks up.

Three separate ways a run looked wrong without being wrong, or looked fine
without being fine:

  - the loss on screen only moved once per epoch, so a healthy run with a few
    batches per epoch read as stalled;
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


class TestLossMovesEveryBatch:
    def test_a_batch_line_updates_the_loss(self, manager):
        job = _Job()
        manager._parse_training_progress(job, """
Starting epoch 1 of 100...
  Batch 1/3 - Loss: 0.233847 (sup: 0.233847, distill: 0.000000)
  Batch 2/3 - Loss: 0.218966 (sup: 0.218809, distill: 0.015784)
""")
        assert job.latest_loss == pytest.approx(0.218966)
        assert (job.current_epoch, job.total_epochs) == (1, 100)
        assert (job.current_batch, job.total_batches) == (2, 3)

    def test_a_batch_loss_is_attributed_to_its_own_epoch(self, manager):
        """The reason per-batch parsing was dropped once.

        Scanned independently, epoch 2's batch loss landed on epoch 1, which
        put the wrong point on the plot. "Starting epoch N" is what keeps the
        two in step.
        """
        job = _Job()
        manager._parse_training_progress(job, """
Starting epoch 1 of 10...
  Batch 1/3 - Loss: 0.9
Epoch 1/10 - Loss: 0.9 - Supervised: 0.9
Starting epoch 2 of 10...
  Batch 1/3 - Loss: 0.4
""")
        assert job.current_epoch == 2
        assert job.latest_loss == pytest.approx(0.4)

    def test_an_epoch_summary_still_wins_when_it_comes_last(self, manager):
        job = _Job()
        manager._parse_training_progress(job, """
Starting epoch 3 of 10...
  Batch 3/3 - Loss: 0.31
Epoch 3/10 - Loss: 0.30 - Supervised: 0.28
""")
        assert job.current_epoch == 3
        assert job.latest_loss == pytest.approx(0.30)

    def test_a_chunk_with_no_starting_line_keeps_the_epoch_it_had(self, manager):
        """The log is read incrementally, so a chunk can be batches only."""
        job = _Job()
        job.current_epoch, job.total_epochs = 7, 10
        manager._parse_training_progress(job, "  Batch 2/3 - Loss: 0.123\n")
        assert job.current_epoch == 7
        assert job.latest_loss == pytest.approx(0.123)


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
