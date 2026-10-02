"""The finetune training job's command line: its flags, and what they describe.

- ``build_arg_parser`` and ``parse_args``: every flag of
  ``python -m cellmap_flow.finetune.finetune_cli``. The job manager builds
  the command, and a running job outlives a dashboard upgrade, so flags only
  ever change compatibly.
- ``model_config_from_args`` and ``build_target_transform``: the model and
  the training target the flags describe.
- ``RESTARTABLE_ARGS`` and ``apply_restart_params``: which settings a
  restart may change, and how a restart request changes them.
"""

import argparse
import json
import logging
import math
from pathlib import Path

from cellmap_flow.finetune.model_loading import decode_model_entry, model_config_from_entry
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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Finetune CellMap-Flow models with LoRA using user corrections"
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="fly",
        choices=["fly", "dacapo", "huggingface", "script", "cellmap", "finetune"],
        help="Model type (fly, dacapo, huggingface, script, cellmap, or finetune). "
             "cellmap takes --model-folder; finetune (continue from a finetuned "
             "model) takes --model-entry."
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

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        # Low rank is itself the anti-forgetting mechanism here: this is
        # correcting a model that is mostly right, so the adapter wants just
        # enough capacity to fix the bad regions and not enough to rewrite
        # the good ones.
        help="LoRA rank (default: 8). 0 = full finetune: every parameter trainable, no adapter; "
             "exports full_finetune/model_state_dict.pt instead of lora_adapter/."
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha scaling (default: twice --lora-r)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)"
    )
    parser.add_argument(
        "--lora-min-channels",
        type=int,
        default=0,
        help="Skip LoRA on layers narrower than this on either side. "
             "Narrow full-resolution layers are where the adapter is expensive "
             "and nearly parameter-free: on mito-aff-unet-setup-16, 96 skips "
             "7 of 19 layers (~1%% of adapter params) for a 1.7x faster step. "
             "(default: 0, adapt every layer)"
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
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation (random flips, XY rotations, brightness "
             "and noise). Patch-center jitter is always applied and is not "
             "affected. Augmentation pays off when a run revisits the same "
             "patches many times; below a few hundred gradient steps it mostly "
             "adds variance, which is why the dashboard defaults it off."
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="combined",
        choices=["dice", "bce", "combined", "mse", "margin", "interval"],
        help="Loss function (default: combined). 'interval' is for a distance model "
             "(--output-type distance) on scribbles: each painted voxel is held between "
             "the bounds the paint implies on its distance to the boundary, exact where "
             "the paint is dense, and the field's slope is limited (--slope-weight)."
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (e.g., 0.1 maps targets from 0/1 to 0.05/0.95). "
             "Helps preserve gradual distance-like outputs. (default: 0.0)"
    )
    parser.add_argument(
        "--distillation-lambda",
        type=float,
        default=None,
        help="Teacher distillation weight. Keeps model close to base on unlabeled voxels. "
             "0.0=disabled, try 0.5-1.0 for sparse scribbles. (default: 1.0 when the "
             "session has good regions, else 0.0; an explicit 0 disables it either way)"
    )
    parser.add_argument(
        "--distillation-all-voxels",
        action="store_true",
        help="Apply distillation loss on all voxels instead of only unlabeled voxels. (default: unlabeled only)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
        help="Margin threshold for margin loss. "
             "Foreground must exceed 1-margin, background must stay below margin. (default: 0.3)"
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Balance fg/bg loss contribution so each class is weighted equally, "
             "regardless of scribble voxel counts. Helps prevent foreground overprediction. (default: off)"
    )
    parser.add_argument(
        "--slope-weight",
        type=float,
        default=1.0,
        help="Weight of --loss-type interval's slope limit, which keeps the predicted "
             "field from getting steeper than a distance field. Without it the bounds "
             "alone let the field collapse into a step. (default: 1.0)"
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision (FP16) training"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers (default: 4)"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Do not write TensorBoard event files to <output-dir>/tensorboard "
             "(default: write them; view with `tensorboard --logdir <training dir>`)"
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
    parser.add_argument(
        "--mask-unannotated",
        action="store_true",
        help="Enable masked loss for sparse annotations (0=ignore, 1=bg, 2+=fg)"
    )

    # Output type and target transform arguments
    parser.add_argument(
        "--output-type",
        type=str,
        default="binary",
        choices=["binary", "binary_broadcast", "affinities", "distance"],
        help="How to generate training targets from annotations. "
             "'binary': single-channel fg/bg (use with --select-channel for multi-channel models). "
             "'binary_broadcast': broadcast binary target to all output channels. "
             "'affinities': compute affinity targets from instance labels (requires offsets). "
             "'distance': soft signed-distance target (tanh(d/sigma)+1)/2 for models trained "
             "the fly_organelles way, e.g. the cellmap *_distance_* repos; requires --loss-type "
             "bce, or interval for scribbles. "
             "(default: binary)"
    )
    parser.add_argument(
        "--distance-sigma",
        type=float,
        default=6.0,
        help="tanh scale in output voxels for --output-type distance. The cellmap "
             "distance models were trained with 6. (default: 6.0)"
    )
    parser.add_argument(
        "--select-channel",
        type=int,
        default=None,
        help="Select a single channel from multi-channel model output for binary training. "
             "Only used with --output-type binary. (default: None, use all channels)"
    )
    parser.add_argument(
        "--offsets",
        type=str,
        default=None,
        help="JSON list of [dz,dy,dx] offsets for affinity target generation. "
             "Example: '[[1,0,0],[0,1,0],[0,0,1]]'. "
             "If not provided with --output-type affinities, will try to read 'offsets' "
             "from the model script."
    )

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """The job's flags, with the defaults that depend on other flags filled in."""
    args = build_arg_parser().parse_args(argv)
    # Keep the LoRA scaling factor (alpha/r) fixed at 2 regardless of rank,
    # which is what FinetuneJobManager already does for dashboard-submitted
    # jobs via lora_alpha = lora_r * 2. A fixed alpha default would silently
    # change the scaling whenever the rank default moved -- at r=64 an
    # alpha of 16 is a scaling of 0.25 rather than 2.
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 2
    return args


def build_target_transform(args, model_config, output_voxel_size_nm=None):
    """Build a TargetTransform based on CLI args.

    ``output_voxel_size_nm`` is the annotation patches' voxel size, which
    the distance bounds of ``--loss-type interval`` are measured in; the
    model's output voxel size when not given.
    """
    from cellmap_flow.finetune.target_transforms import (
        BinaryTargetTransform,
        BroadcastBinaryTargetTransform,
        AffinityTargetTransform,
        DistanceTargetTransform,
        IntervalTargetTransform,
    )

    output_type = args.output_type
    num_channels = model_config.config.output_channels

    if getattr(args, "loss_type", None) == "interval" and output_type != "distance":
        raise ValueError(
            "--loss-type interval bounds a distance model's output; it needs "
            "--output-type distance."
        )

    # --select-channel slices the prediction to one channel, so the target
    # must have one too. Distance and binary_broadcast built theirs with every
    # channel, and the first batch failed on the size mismatch.
    select_channel = getattr(args, "select_channel", None)
    if select_channel is not None:
        if not 0 <= int(select_channel) < num_channels:
            raise ValueError(
                f"--select-channel {select_channel} is out of range for a model with "
                f"{num_channels} output channel(s)."
            )
        if output_type == "affinities":
            raise ValueError(
                "--select-channel cannot be combined with --output-type affinities: the "
                "affinity target has one channel per offset. Drop --select-channel, or "
                "train that channel with --output-type binary."
            )
        num_channels = 1

    if output_type == "binary":
        if num_channels > 1:
            if getattr(args, "loss_type", None) in ("bce", "mse", "combined"):
                # BCE compares shapes exactly: this failed on the first batch
                # with "Target size [2, 1, ...] must be the same as input size
                # [2, 3, ...]". Dice and margin broadcast the one-channel
                # target over the channels, and keep doing so.
                raise ValueError(
                    f"The model has {num_channels} output channels but --output-type "
                    f"binary makes a one-channel target, which --loss-type "
                    f"{args.loss_type} cannot compare with. Train one channel with "
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

        # Try CLI arg first
        if args.offsets:
            offsets = json.loads(args.offsets)

        # Try reading from model script
        if offsets is None and args.model_script:
            offsets = read_offsets_from_script(args.model_script)

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

    elif output_type == "distance" and args.loss_type == "interval":
        if output_voxel_size_nm is None:
            output_voxel_size_nm = getattr(model_config.config, "output_voxel_size", None)
        logger.info(
            f"Using distance bounds from the paint (sigma={args.distance_sigma} voxels, "
            f"voxel size {output_voxel_size_nm} nm, slope weight {args.slope_weight})"
        )
        return IntervalTargetTransform(args.distance_sigma, voxel_size_nm=output_voxel_size_nm)

    elif output_type == "distance":
        if args.loss_type != "bce":
            raise ValueError(
                "--output-type distance produces soft targets in [0, 1]; only "
                "--loss-type bce (BCE with logits) is supported for them, or "
                "interval for scribbles. Margin and dice assume hard labels."
            )
        if args.label_smoothing > 0:
            logger.warning(
                "Label smoothing is meaningless on a soft distance target; "
                f"ignoring --label-smoothing {args.label_smoothing}."
            )
            args.label_smoothing = 0.0
        if getattr(args, "mask_unannotated", False):
            logger.warning(
                "--output-type distance with --mask-unannotated (sparse/scribble "
                "annotations): a distance transform needs dense 3D labels, and "
                "voxels next to unannotated ones are left out of the loss, so "
                "very little of a scribble session will be supervised. Use "
                "--loss-type interval for scribbles."
            )
        logger.info(
            f"Using distance target transform (sigma={args.distance_sigma} voxels, "
            f"broadcast to {num_channels} channel(s))"
        )
        return DistanceTargetTransform(args.distance_sigma, num_channels=num_channels)

    else:
        raise ValueError(f"Unknown output type: {output_type}")


def model_config_from_args(args) -> ModelConfig:
    """The ModelConfig the command line describes."""
    if args.model_entry:
        # The model's own to_dict(), as the job manager passes it for the
        # types whose flags cannot say all of it (cellmap, finetune, fly;
        # job_manager.submit.MODEL_ENTRY_TYPES).
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
        # No flag gives its input and output sizes, so this is
        # fly_organelles' StandardUnet (178 in, 56 out). The job manager
        # sends a Fly model as --model-entry instead, sizes and all.
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


# What a restart may change: training settings only. The model, the data and
# every path stay as launched, so a restart request cannot point the job at
# other files. Matches the dashboard's RESTART_PASSTHROUGH_KEYS, plus the
# distillation_all_voxels flag it derives from distillation_scope, less
# patch_shape: the patch geometry comes from the corrections' manifest, and
# nothing reads --patch-shape.
RESTARTABLE_ARGS = frozenset(
    {
        "lora_r", "lora_alpha", "num_epochs", "batch_size", "learning_rate",
        "loss_type", "label_smoothing", "distillation_lambda",
        "distillation_all_voxels", "margin", "balance_classes", "augment",
        "mask_unannotated", "gradient_accumulation_steps", "num_workers",
        "no_augment", "no_mixed_precision", "output_type",
        "select_channel", "offsets",
    }
)


def _as_bool(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off", ""):
            return False
        raise ValueError(f"not a boolean: {value!r}")
    return bool(value)


def _at_least(minimum):
    def convert(value):
        number = int(value)
        if number < minimum:
            raise ValueError(f"must be at least {minimum}")
        return number
    return convert


def _positive(value):
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError("must be a number above 0")
    return number


def _is_offset(offset):
    """A [z, y, x] of whole numbers."""
    return (
        isinstance(offset, list)
        and len(offset) == 3
        and all(isinstance(v, int) and not isinstance(v, bool) for v in offset)
    )


def _as_offsets(value):
    # --offsets is a JSON string. A restart may carry the list itself, which
    # build_target_transform's json.loads() would then reject.
    offsets = json.loads(value) if isinstance(value, str) else value
    if not (isinstance(offsets, list) and offsets and all(_is_offset(o) for o in offsets)):
        raise ValueError("expected a list of [z, y, x] integer offsets")
    return json.dumps(offsets)


def _one_of(*choices):
    def convert(value):
        if value not in choices:
            raise ValueError(f"{value!r} is not one of {list(choices)}")
        return value
    return convert


# How each restartable setting is read: the argparse destination it sets and
# the conversion applied to the requested value. Most keys are the
# destination itself. The dashboard's "augment" is the inverse of the CLI's
# --no-augment, and used to be dropped by a hasattr(args, key) filter, so a
# restart could never switch augmentation on or off.
#
# A value that converts but cannot train is refused here, so that the job
# keeps the setting it has. Found only once training ran, it failed a job
# that was serving: 0 epochs exported the previous iteration's best
# checkpoint, 0 accumulation steps was reported as divergence, two-value
# offsets failed at the first batch.
_RESTART_ARG_CONVERTERS = {
    "lora_r": ("lora_r", int),
    "lora_alpha": ("lora_alpha", int),
    "num_epochs": ("num_epochs", _at_least(1)),
    "batch_size": ("batch_size", _at_least(1)),
    "learning_rate": ("learning_rate", _positive),
    "loss_type": ("loss_type", _one_of("dice", "bce", "combined", "mse", "margin", "interval")),
    "label_smoothing": ("label_smoothing", float),
    "distillation_lambda": ("distillation_lambda", float),
    "distillation_all_voxels": ("distillation_all_voxels", _as_bool),
    "margin": ("margin", float),
    "balance_classes": ("balance_classes", _as_bool),
    "augment": ("no_augment", lambda value: not _as_bool(value)),
    "mask_unannotated": ("mask_unannotated", _as_bool),
    "gradient_accumulation_steps": ("gradient_accumulation_steps", _at_least(1)),
    "num_workers": ("num_workers", _at_least(0)),
    "no_augment": ("no_augment", _as_bool),
    "no_mixed_precision": ("no_mixed_precision", _as_bool),
    "output_type": (
        "output_type", _one_of("binary", "binary_broadcast", "affinities", "distance")
    ),
    "select_channel": ("select_channel", int),
    "offsets": ("offsets", _as_offsets),
}
assert set(_RESTART_ARG_CONVERTERS) == RESTARTABLE_ARGS


def apply_restart_params(args, signal_data: dict):
    """
    Update args with parameters from restart signal and persist to metadata.json.

    Args:
        args: argparse Namespace to update
        signal_data: Dict from restart signal file
    """
    params = signal_data.get("params", {}) or {}
    refused = sorted(set(params) - RESTARTABLE_ARGS)
    if refused:
        logger.warning(f"Ignoring restart parameters that restarts cannot change: {refused}")
    changed = False
    # What was applied, under the name the request used, for metadata.json.
    # Refused keys reach neither args nor metadata.json.
    recorded = {}
    for key, value in params.items():
        if key not in RESTARTABLE_ARGS or value is None:
            continue
        dest, convert = _RESTART_ARG_CONVERTERS[key]
        if not hasattr(args, dest):
            logger.warning(f"Restart parameter {key!r} has no matching training setting; ignoring it.")
            continue
        try:
            new_value = convert(value)
        except (TypeError, ValueError) as e:
            logger.warning(f"Ignoring restart parameter {key}={value!r}: {e}")
            continue
        old_value = getattr(args, dest)
        setattr(args, dest, new_value)
        recorded[key] = _as_bool(value) if key == "augment" else new_value
        if old_value != new_value:
            logger.info(f"Updated {dest}: {old_value} -> {new_value}")
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
