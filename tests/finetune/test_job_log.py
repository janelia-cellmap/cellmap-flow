"""The trainer's log as the job manager reads it: whole lines, and the model and
serving YAML of the last iteration. A training job outlives a dashboard
upgrade, so both sides of these lines are a protocol."""

from cellmap_flow.finetune import markers
from cellmap_flow.finetune.finetune_job_manager import trainer_outputs_from_log
from cellmap_flow.finetune.job_log import LogTailer


def test_the_tailer_hands_out_whole_lines_and_starts_over_on_a_new_file(tmp_path):
    log = tmp_path / "training_log.txt"
    log.write_text("Starting epoch 3 of 10...\nEpoch 3/10 - Lo")
    tail = LogTailer(log)

    assert tail.read() == "Starting epoch 3 of 10...\n"
    with open(log, "a") as f:
        f.write("ss: 0.25\nTRAINING_ITERATION_COM")
    assert tail.read() == "Epoch 3/10 - Loss: 0.25\n"
    log.write_text("new\n")  # shorter: replaced, not appended to
    assert tail.read() == "new\n", "the held-back half line belonged to the old file"


def test_the_last_iteration_the_trainer_announced(capsys):
    markers.emit(markers.FINETUNED_MODEL_YAML, "/s/models/m_1.yaml")
    markers.emit(markers.TRAINING_ITERATION_COMPLETE, "m_1")
    markers.emit(markers.WAITING_FOR_RESTART)
    log = capsys.readouterr().out

    assert trainer_outputs_from_log(log) == ("m_1", "/s/models/m_1.yaml")
    # An iteration whose YAML could not be written is not given the previous one's.
    assert trainer_outputs_from_log(log + "TRAINING_ITERATION_COMPLETE: m_2\n") == ("m_2", None)
    assert trainer_outputs_from_log("Starting epoch 1 of 5...\n") == (None, None)
