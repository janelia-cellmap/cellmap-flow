"""What a restart request changes in the running trainer.

The dashboard sends "augment", the trainer's flag is --no-augment, and
_apply_restart_params only set keys argparse already had -- so the toggle in
the restart dialog was dropped without a word. Offsets sent as a list would
likewise have reached json.loads() and killed the restart.
"""

import argparse
import json

from cellmap_flow.finetune.finetune_cli import _apply_restart_params


def _args(tmp_path=None, **kw):
    base = dict(
        output_dir=str(tmp_path) if tmp_path else None,
        no_augment=True, offsets=None, learning_rate=1e-4, loss_type="bce",
        lora_r=8, lora_alpha=16, balance_classes=False, num_epochs=10,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_augment_turns_augmentation_on_and_off():
    args = _args()
    _apply_restart_params(args, {"params": {"augment": True}})
    assert args.no_augment is False
    _apply_restart_params(args, {"params": {"augment": False}})
    assert args.no_augment is True


def test_augment_is_recorded_under_its_own_name(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"params": {"augment": False}}))
    _apply_restart_params(_args(tmp_path), {"params": {"augment": True}})
    assert json.loads((tmp_path / "metadata.json").read_text())["params"]["augment"] is True


def test_offsets_given_as_a_list_become_the_json_the_trainer_reads():
    args = _args()
    _apply_restart_params(args, {"params": {"offsets": [[1, 0, 0], [0, 1, 0]]}})
    assert json.loads(args.offsets) == [[1, 0, 0], [0, 1, 0]]


def test_values_are_converted_to_the_settings_types():
    args = _args()
    _apply_restart_params(
        args,
        {"params": {"learning_rate": "5e-4", "num_epochs": "20", "balance_classes": "true"}},
    )
    assert args.learning_rate == 5e-4
    assert args.num_epochs == 20
    assert args.balance_classes is True


def test_a_value_that_cannot_be_used_is_ignored_not_applied():
    args = _args()
    _apply_restart_params(
        args, {"params": {"loss_type": "hinge", "num_epochs": "many", "learning_rate": 3e-4}}
    )
    assert args.loss_type == "bce"
    assert args.num_epochs == 10
    assert args.learning_rate == 3e-4, "the valid settings in the same request still apply"


def test_the_dashboard_sends_the_trainers_own_flag_and_json_offsets():
    from cellmap_flow.dashboard.routes.finetune.common import build_restart_params

    params = build_restart_params({"augment": True, "offsets": [[1, 0, 0]]})
    assert params["augment"] is True
    assert params["no_augment"] is False
    assert json.loads(params["offsets"]) == [[1, 0, 0]]
