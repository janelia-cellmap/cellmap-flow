"""The finetune training job's command line: its flags, and what they describe.

- ``build_arg_parser`` and ``parse_args``: every flag of
  ``python -m cellmap_flow.finetune.finetune_cli``. The training settings'
  flags are made from ``options.TrainingOptions``, one per field, and
  ``options_of`` reads them back into one; ``training_flags`` is the
  inverse, which the job manager builds the command with. A running job
  outlives a dashboard upgrade, so flags only ever change compatibly.
- ``model_config_from_args`` and ``build_target_transform``: the model and
  the training target the flags describe.
- ``RESTARTABLE_ARGS`` and ``apply_restart_params``: which settings a
  restart may change, and how a restart request changes them.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Literal, Union, get_args, get_origin

from pydantic import TypeAdapter, ValidationError

from cellmap_flow.finetune.job_manager.submit import trainable_model_types
from cellmap_flow.finetune.model_loading import decode_model_entry, model_config_from_entry
from cellmap_flow.finetune.options import MANIFEST_FIELDS, TrainingOptions
from cellmap_flow.finetune.target_transforms import read_offsets_from_script
from cellmap_flow.jobs.site import current_site
from cellmap_flow.models.models_config import (
    DaCapoModelConfig,
    FlyModelConfig,
    HuggingFaceModelConfig,
    ModelConfig,
)
from cellmap_flow.finetune.json_files import write_json_atomically

logger = logging.getLogger(__name__)


def _flag_type(annotation):
    """(type, choices) of a field's flag: Optional[X] is X, a Literal is str with its values as the choices."""
    if get_origin(annotation) is Union:  # Optional[X]
        annotation = next(a for a in get_args(annotation) if a is not type(None))
    if get_origin(annotation) is Literal:
        return str, get_args(annotation)
    return annotation, None


def add_training_flags(parser):
    """A flag per ``TrainingOptions`` field: ``--<name>`` with the field's default and its
    description as help, ``--<name>``/``--no-<name>`` for a boolean. The manifest's fields
    have none (see options.MANIFEST_FIELDS)."""
    group = parser.add_argument_group("training settings (finetune/options.py)")
    for name, field in TrainingOptions.model_fields.items():
        if name in MANIFEST_FIELDS:
            continue
        flag = "--" + name.replace("_", "-")
        kind, choices = _flag_type(field.annotation)
        help_text = (field.description or "").replace("%", "%%")
        if kind is bool:
            group.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default, help=help_text)
        else:
            if field.default is not None:
                help_text += f" (default: {field.default})"
            group.add_argument(flag, type=kind, default=field.default, choices=choices, help=help_text)
    # The flag jobs were started with before --distillation-scope: the same setting.
    group.add_argument(
        "--distillation-all-voxels", action="store_const", const="all", dest="distillation_scope",
        help="The same as --distillation-scope all.",
    )


def options_of(args) -> TrainingOptions:
    """The training settings the parsed command line ``args`` carries."""
    return TrainingOptions.model_validate(
        {name: getattr(args, name) for name in TrainingOptions.model_fields if hasattr(args, name)}
    )


def training_flags(options: TrainingOptions) -> list:
    """``options`` as the flags ``build_arg_parser`` takes, every setting said so the command
    stands on its own: ``--<name> <value>``, ``--<name>`` or ``--no-<name>`` for a boolean;
    a blank value and the manifest's fields have no flag."""
    flags = []
    for name, value in options.model_dump(exclude=set(MANIFEST_FIELDS)).items():
        dashed = name.replace("_", "-")
        if isinstance(value, bool):
            flags.append(f"--{dashed}" if value else f"--no-{dashed}")
        elif value is not None:
            flags += [f"--{dashed}", str(value)]
    return flags


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Finetune a CellMap-Flow model on user corrections: a LoRA adapter, or every weight"
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="fly",
        choices=trainable_model_types(),
        help="Model type: one of the types that can be finetuned. dacapo, "
             "huggingface and script take their own flags; the others take "
             "--model-entry (the model's entry, as JSON)."
    )
    parser.add_argument(
        "--model-entry",
        type=str,
        default=None,
        help="The model as its model entry (what ModelConfig.to_dict() gives, "
             "the same shape as a models: entry in a YAML), as JSON or "
             "encode_to_str()'d JSON. Takes precedence over the other model flags."
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        default=None,
        help="Folder of a cellmap model (for --model-type cellmap)"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to model checkpoint (optional - can train from scratch)"
    )
    parser.add_argument(
        "--model-script",
        type=str,
        required=False,
        default=None,
        help="Path to model script (alternative to checkpoint)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=False,
        default=None,
        help="HuggingFace model repository (e.g., janelia-cellmap/mito_aff_unet_setup_16)"
    )
    parser.add_argument(
        "--revision",
        type=str,
        required=False,
        default=None,
        help="HuggingFace model revision (optional)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (for filtering corrections)"
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=["mito"],
        help="Model output channels"
    )
    parser.add_argument(
        "--input-voxel-size",
        type=int,
        nargs=3,
        default=[16, 16, 16],
        help="Input voxel size (Z Y X)"
    )
    parser.add_argument(
        "--output-voxel-size",
        type=int,
        nargs=3,
        default=[16, 16, 16],
        help="Output voxel size (Z Y X)"
    )

    # Data arguments
    parser.add_argument(
        "--corrections",
        type=str,
        required=True,
        help="Path to corrections.zarr directory"
    )
    parser.add_argument(
        "--patch-shape",
        type=int,
        nargs=3,
        default=None,
        help="Unused, and accepted so that older commands still parse: the "
             "patch shape comes from the corrections' manifest."
    )
    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and adapter"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory for the generated serving YAMLs (default: <session>/models "
             "when --output-dir is <session>/runs/<name>, else <output-dir>/models)"
    )
    parser.add_argument(
        "--queue",
        type=str,
        default=None,
        help="LSF queue written into the generated serving YAMLs "
             f"(default: {current_site().default_queue})"
    )
    parser.add_argument(
        "--charge-group",
        type=str,
        default=None,
        help="LSF charge group written into the generated serving YAMLs "
             f"(default: {current_site().default_charge_group})"
    )
    # Resuming
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    # Auto-serve arguments
    parser.add_argument(
        "--auto-serve",
        action="store_true",
        help="Automatically start inference server after training completes"
    )
    parser.add_argument(
        "--serve-data-path",
        type=str,
        default=None,
        help="Dataset path for inference server (required if --auto-serve is used)"
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=0,
        help="Port for inference server (0 for auto-assignment)"
    )
    add_training_flags(parser)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """The job's flags, the training settings checked and completed as TrainingOptions
    checks them (a rank of 0 is mode full; alpha is twice the rank unless given)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        options = options_of(args)
    except ValidationError as e:
        parser.error(_reason(e))
    for name in TrainingOptions.model_fields:
        if hasattr(args, name):
            setattr(args, name, getattr(options, name))
    return args


def _reason(error) -> str:
    """A ValidationError's first message, with the field it is about."""
    if isinstance(error, ValidationError):
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        message = first["msg"]
        if "error" in first.get("ctx", {}):
            message = str(first["ctx"]["error"])
        return f"{where}: {message}" if where else message
    return str(error)


def build_target_transform(options: TrainingOptions, model_config, output_voxel_size_nm=None, script=None):
    """The TargetTransform ``options`` describe (output_type, loss_type, select_channel, offsets).

    ``output_voxel_size_nm`` is the annotation patches' voxel size, which
    the distance bounds of ``--loss-type interval`` are measured in; the
    model's output voxel size when not given. ``script`` is the model script
    an affinity model's offsets are read from when the options name none;
    the model config's own otherwise.
    """
    from cellmap_flow.finetune.target_transforms import (
        BinaryTargetTransform,
        BroadcastBinaryTargetTransform,
        AffinityTargetTransform,
        DistanceTargetTransform,
        IntervalTargetTransform,
    )

    output_type = options.output_type
    loss_type = options.loss_type
    num_channels = model_config.config.output_channels

    if loss_type == "interval" and output_type != "distance":
        raise ValueError(
            "--loss-type interval bounds a distance model's output; it needs "
            "--output-type distance."
        )
    # The flow target and the flow loss only make sense together: the target
    # is flows and a foreground channel, which no other loss reads right, and
    # the loss needs its per-channel masks.
    if (loss_type == "flow") != (output_type == "flows"):
        raise ValueError(
            "--output-type flows and --loss-type flow go together: the flow target is "
            "flows towards each instance's centre and a foreground channel, which only "
            "the flow loss compares with."
        )

    # --select-channel slices the prediction to one channel, so the target
    # must have one too. Distance and binary_broadcast built theirs with every
    # channel, and the first batch failed on the size mismatch.
    select_channel = options.select_channel
    if select_channel is not None:
        if not 0 <= int(select_channel) < num_channels:
            raise ValueError(
                f"--select-channel {select_channel} is out of range for a model with "
                f"{num_channels} output channel(s)."
            )
        if output_type in ("affinities", "flows"):
            raise ValueError(
                f"--select-channel cannot be combined with --output-type {output_type}: the "
                "target has a channel of its own for each of the model's. Drop "
                "--select-channel, or train that channel with --output-type binary."
            )
        num_channels = 1

    if output_type == "binary":
        if num_channels > 1:
            if loss_type in ("bce", "mse", "combined"):
                # BCE compares shapes exactly: this failed on the first batch
                # with "Target size [2, 1, ...] must be the same as input size
                # [2, 3, ...]". Dice and margin broadcast the one-channel
                # target over the channels, and keep doing so.
                raise ValueError(
                    f"The model has {num_channels} output channels but --output-type "
                    f"binary makes a one-channel target, which --loss-type "
                    f"{loss_type} cannot compare with. Train one channel with "
                    f"--select-channel, or all of them with --output-type binary_broadcast."
                )
            logger.warning(
                f"Model has {num_channels} output channels but --output-type is 'binary' "
                f"and --select-channel is not set. Consider using --select-channel or "
                f"--output-type binary_broadcast."
            )
        return BinaryTargetTransform()

    elif output_type == "binary_broadcast":
        logger.info(f"Broadcasting binary target to {num_channels} channels")
        return BroadcastBinaryTargetTransform(num_channels)

    elif output_type == "affinities":
        offsets = None

        if options.offsets:
            offsets = json.loads(options.offsets)

        # Else from the model script (a model with an env comes as
        # --model-entry, its script path in the config)
        script = script or getattr(model_config, "script_path", None)
        if offsets is None and script:
            offsets = read_offsets_from_script(script)

        if offsets is None:
            raise ValueError(
                "Affinity output type requires offsets. Provide --offsets as a JSON list "
                "(e.g. '[[1,0,0],[0,1,0],[0,0,1]]') or define an 'offsets' variable in "
                "the model script."
            )

        if len(offsets) > num_channels:
            raise ValueError(
                f"Number of offsets ({len(offsets)}) exceeds model output channels "
                f"({num_channels})."
            )

        if len(offsets) < num_channels:
            logger.info(
                f"Model has {num_channels} output channels but only {len(offsets)} affinity offsets. "
                f"Remaining {num_channels - len(offsets)} channels (e.g. LSDs) will be masked out."
            )

        logger.info(f"Using affinity target transform with {len(offsets)} offsets: {offsets}")
        return AffinityTargetTransform(offsets, num_channels=num_channels)

    elif output_type == "distance" and loss_type == "interval":
        if output_voxel_size_nm is None:
            output_voxel_size_nm = getattr(model_config.config, "output_voxel_size", None)
        logger.info(
            f"Using distance bounds from the paint (sigma={options.distance_sigma} voxels, "
            f"voxel size {output_voxel_size_nm} nm, slope weight {options.slope_weight})"
        )
        return IntervalTargetTransform(options.distance_sigma, voxel_size_nm=output_voxel_size_nm)

    elif output_type == "distance":
        if loss_type != "bce":
            raise ValueError(
                "--output-type distance produces soft targets in [0, 1]; only "
                "--loss-type bce (BCE with logits) is supported for them, or "
                "interval for scribbles. Margin and dice assume hard labels."
            )
        if options.mask_unannotated:
            logger.warning(
                "--output-type distance with --mask-unannotated (sparse/scribble "
                "annotations): a distance transform needs dense 3D labels, and "
                "voxels next to unannotated ones are left out of the loss, so "
                "very little of a scribble session will be supervised. Use "
                "--loss-type interval for scribbles."
            )
        logger.info(
            f"Using distance target transform (sigma={options.distance_sigma} voxels, "
            f"broadcast to {num_channels} channel(s))"
        )
        return DistanceTargetTransform(options.distance_sigma, num_channels=num_channels)

    elif output_type == "flows":
        from cellmap_flow.finetune.instance_flows import FlowTargetTransform

        logger.info(
            "Using flow targets: per-slice flows towards each painted instance's centre "
            "and a foreground channel (instances cut by the patch edge leave the flow loss)"
        )
        return FlowTargetTransform()

    else:
        raise ValueError(f"Unknown output type: {output_type}")


def model_config_from_args(args) -> ModelConfig:
    """The ModelConfig the command line describes."""
    if args.model_entry:
        # The model's own to_dict(), as the job manager passes it for every
        # type without flags of its own (all but job_manager.submit's
        # FLAG_MODEL_TYPES).
        entry = decode_model_entry(args.model_entry)
        logger.info(f"Using model entry of type {entry.get('type')!r}")
        return model_config_from_entry(entry, name=args.model_name)
    if args.model_script:
        from cellmap_flow.models.models_config import ScriptModelConfig
        logger.info(f"Using script-based model: {args.model_script}")
        return ScriptModelConfig(
            script_path=args.model_script,
            name=args.model_name or "script_model"
        )
    if args.model_type == "script":
        raise ValueError("For script models, --model-script is required")
    if args.model_type == "fly":
        if not args.model_checkpoint:
            raise ValueError(
                "For fly models, either --model-checkpoint or --model-script must be provided"
            )
        # No flag gives its input and output sizes, so they are what the
        # checkpoint's folder says, else 178 in and what the network makes
        # of it. The job manager sends a Fly model as --model-entry instead,
        # sizes and all.
        return FlyModelConfig(
            checkpoint_path=args.model_checkpoint,
            channels=args.channels,
            input_voxel_size=tuple(args.input_voxel_size),
            output_voxel_size=tuple(args.output_voxel_size),
            name=args.model_name,
        )
    if args.model_type == "dacapo":
        if not args.model_checkpoint:
            raise ValueError("For dacapo models, --model-checkpoint is required")
        checkpoint_path = Path(args.model_checkpoint)
        iteration = int(checkpoint_path.stem.split('_')[-1])
        run_name = checkpoint_path.parent.name
        return DaCapoModelConfig(
            run_name=run_name,
            iteration=iteration,
        )
    if args.model_type == "huggingface":
        if not args.repo:
            raise ValueError("For huggingface models, --repo is required")
        return HuggingFaceModelConfig(
            repo=args.repo,
            revision=args.revision,
            name=args.model_name,
        )
    if args.model_type == "cellmap":
        if not args.model_folder:
            raise ValueError("For cellmap models, --model-folder (or --model-entry) is required")
        from cellmap_flow.models.models_config import CellMapModelConfig
        return CellMapModelConfig(folder_path=args.model_folder, name=args.model_name)
    if args.model_type == "finetune":
        raise ValueError(
            "For finetune models, --model-entry is required: the model's entry "
            "(type: finetune, base_model, lora_adapter_path or weights_path) as JSON"
        )
    raise ValueError(f"Unknown model type: {args.model_type}")


_BOOL = TypeAdapter(bool)

# The dashboard's names for three settings the CLI once called differently,
# and the setting each sets: a dashboard restarts jobs started before, so it
# goes on sending these (common.restart_params), and the CLI goes on
# taking them.
RESTART_ALIASES = {
    "no_augment": ("augment", lambda value: not _BOOL.validate_python(value)),
    "no_mixed_precision": ("mixed_precision", lambda value: not _BOOL.validate_python(value)),
    "distillation_all_voxels": ("distillation_scope", lambda value: "all" if _BOOL.validate_python(value) else "unlabeled"),
}

# What a restart may change: training settings only. The model, the data and
# every path stay as launched, so a restart request cannot point the job at
# other files; nor can it change the mode or the output class, which are the
# model's (the inference server shares it, and it is built once), or the
# manifest's fields, which the dashboard writes into the manifest instead.
RESTARTABLE_ARGS = frozenset(
    (set(TrainingOptions.model_fields) - {"mode", "output_class", *MANIFEST_FIELDS}) | set(RESTART_ALIASES)
)


def apply_restart_params(args, signal_data: dict):
    """
    Update args with parameters from restart signal and persist to metadata.json.

    Each value is checked as TrainingOptions checks it, so one that cannot
    train (0 epochs, a learning rate of 0, two-value offsets) is refused
    here, with its reason, and the job keeps the setting it has; found only
    once training ran, it failed a job that was serving. A rank for a full
    finetune, or a rank of 0 for a LoRA job, is refused too: the model is
    built once, so a job never switches between the two.

    Args:
        args: argparse Namespace to update
        signal_data: Dict from restart signal file
    """
    params = signal_data.get("params", {}) or {}
    refused = sorted(set(params) - RESTARTABLE_ARGS)
    if refused:
        logger.warning(f"Ignoring restart parameters that restarts cannot change: {refused}")
    options = options_of(args)
    changed = False
    # What was applied, under the name the request used, for metadata.json.
    # Refused keys reach neither args nor metadata.json.
    recorded = {}
    for key, value in params.items():
        if key not in RESTARTABLE_ARGS or value is None:
            continue
        field, convert = RESTART_ALIASES.get(key, (key, lambda v: v))
        if not hasattr(args, field):
            logger.warning(f"Restart parameter {key!r} has no matching training setting; ignoring it.")
            continue
        try:
            requested = convert(value)
            updated = options.replaced(**{field: requested})
        except (ValidationError, TypeError, ValueError) as e:
            logger.warning(f"Ignoring restart parameter {key}={value!r}: {_reason(e)}")
            continue
        if field in ("lora_r", "lora_alpha") and options.mode == "full":
            logger.warning(f"Ignoring restart parameter {key}={value!r}: this job is a full finetune, "
                           "which has no adapter; submit a new job for a LoRA adapter.")
            continue
        if field == "lora_r" and updated.mode == "full":
            logger.warning("Restart asked for lora_r=0 (full finetune) but this job trains a LoRA adapter; "
                           "submit a new job for that. Keeping the current adapter setup.")
            continue
        old_value, new_value = getattr(options, field), getattr(updated, field)
        options = updated
        setattr(args, field, new_value)
        recorded[key] = _BOOL.validate_python(value) if key in RESTART_ALIASES else new_value
        if old_value != new_value:
            logger.info(f"Updated {field}: {old_value} -> {new_value}")
            changed = True

    # alpha is what sets LoRA's step size: peft scales the adapter by
    # lora_alpha / r. Submit derives alpha = 2 * r, but a restart only carries
    # lora_r -- so raising the rank from 8 to 64 while alpha stayed at 16 cut
    # the effective update to an eighth, and produced a loss curve that looks
    # reassuringly smooth because very little is happening per step.
    if "lora_r" in recorded and "lora_alpha" not in recorded:
        derived = int(recorded["lora_r"]) * 2
        if getattr(args, "lora_alpha", None) != derived:
            logger.info(
                f"Updated lora_alpha: {getattr(args, 'lora_alpha', None)} -> "
                f"{derived} (held at 2x rank so the adapter scaling does not "
                f"change when you change the rank)"
            )
            args.lora_alpha = derived
            recorded["lora_alpha"] = derived
            changed = True

    # Persist updated params to metadata.json
    if changed and getattr(args, "output_dir", None):
        def record(metadata):
            if "params" in metadata:
                for key, value in recorded.items():
                    if key in metadata["params"]:
                        metadata["params"][key] = value
            metadata["last_restart_at"] = signal_data.get("timestamp")

        try:
            if update_run_metadata(args.output_dir, record):
                logger.info("Updated metadata.json with restart params")
        except Exception as e:
            logger.warning(f"Failed to update metadata.json: {e}")


def update_run_metadata(output_dir, change) -> bool:
    """Apply ``change(metadata)`` to the run's metadata.json; False if the run has none.

    The dashboard's job monitor updates the same file from its own host,
    within seconds of every restart. The file is read just before it is
    replaced, and replaced whole (write_json_atomically), so neither side
    can read the other's write half done, and the window in which both
    update it at once, and one loses the other's change, is kept short.
    A run that the job manager did not start has no metadata.json, and gets
    none.
    """
    path = Path(output_dir) / "metadata.json"
    if not path.exists():
        return False
    metadata = json.loads(path.read_text())
    change(metadata)
    write_json_atomically(path, metadata)
    return True
