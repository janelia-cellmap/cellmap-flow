"""What a submission runs, and where.

The manager's submit_finetuning_job validates a request and writes the
run's directory; this module answers the questions on the way:

- ``resolve_model_type``: the trainer's ``--model-type`` for a model
  config, refusing one it cannot train;
- ``find_checkpoint``, ``count_corrections``: the checkpoint to start from,
  and whether the session has anything to train on;
- ``model_settings``: the channels and voxel sizes to train with;
- ``extract_data_path_from_corrections``: the raw data the job's server
  serves;
- ``build_command``: the trainer's shell command line, which LSF runs on a
  GPU node (``python -m cellmap_flow.finetune.finetune_cli``);
- ``launch``: that command on LSF, or here when there is no bsub.
"""

import functools
import json
import logging
import os
import string
import sys
from pathlib import Path
from typing import List, Optional

from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.jobs.site import current_site
from cellmap_flow.jobs.spec import JobSpec
# Module globals, looked up when a job is submitted, so tests can replace
# them here.
from cellmap_flow.jobs.lsf import available as is_bsub_available
from cellmap_flow.jobs.local import run as run_locally

logger = logging.getLogger(__name__)


# The --model-type values finetune_cli accepts, and the ones among them that
# it takes as --model-entry (the model's to_dict()) because they have no
# dedicated flags.
TRAINABLE_MODEL_TYPES = frozenset({"fly", "dacapo", "huggingface", "script", "cellmap", "finetune"})
MODEL_ENTRY_TYPES = frozenset({"cellmap", "finetune"})


def resolve_model_type(model_config) -> str:
    """Infer the finetuning CLI model type from the model config.

    Raises ValueError for a type the trainer cannot train, rather than
    submitting a GPU job whose argparse exits with code 2.
    """
    model_type = getattr(type(model_config), "cli_name", "fly")
    if model_type == "fly" and "dacapo" in model_config.name.lower():
        return "dacapo"
    if model_type not in TRAINABLE_MODEL_TYPES:
        raise ValueError(
            f"Models of type {model_type!r} cannot be finetuned; the trainer "
            f"supports {sorted(TRAINABLE_MODEL_TYPES)}."
        )
    return model_type


def find_checkpoint(model_config, override=None) -> Optional[Path]:
    """The checkpoint to finetune from: ``override``, else the config's own.

    None when there is neither: a script model passes its script instead.
    Raises ValueError for a checkpoint that does not exist.
    """
    checkpoint_path = None

    # Check for checkpoint override first
    if override:
        checkpoint_path = Path(override)
        logger.info(f"Using checkpoint path override: {checkpoint_path}")
    # For FlyModelConfig, get checkpoint_path attribute
    elif hasattr(model_config, 'checkpoint_path') and model_config.checkpoint_path:
        checkpoint_path = Path(model_config.checkpoint_path)
        logger.info(f"Found checkpoint_path: {checkpoint_path}")

    # Validate checkpoint exists if specified
    if checkpoint_path and not checkpoint_path.exists():
        raise ValueError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            f"Please verify the path exists and is accessible."
        )
    return checkpoint_path


def count_corrections(corrections_path: Path) -> int:
    """How many corrections the session holds.

    Raises ValueError when there is nothing to train on.
    """
    if not corrections_path.exists():
        raise ValueError(f"Corrections path does not exist: {corrections_path}")

    # The trainer reads the session's virtual-sources manifest and nothing
    # else, so a session without one failed on the GPU node with
    # FileNotFoundError after queueing. This used to count *.zarr
    # directories instead: always "Only 1 corrections" for a volume
    # session, and any crop zarr passed.
    from cellmap_flow.finetune.session.manifest import VIRTUAL_MANIFEST_FILENAME, read_manifest

    if read_manifest(str(corrections_path)) is None:
        raise ValueError(
            f"No {VIRTUAL_MANIFEST_FILENAME} in {corrections_path}, so there is "
            "nothing to train on. Create an annotation volume, or import crops, first."
        )
    correction_dirs = list(corrections_path.glob("*/"))
    return len([d for d in correction_dirs if (d / ".zattrs").exists()])


def model_settings(model_config, warned: set):
    """(channels, input voxel size, output voxel size) to train ``model_config`` with.

    Each is the config's own, else its geometry's, else a guess: channels
    ["mito"], 16 nm voxels. A guess is logged once per model; ``warned``
    holds the names of the models already warned about.
    """
    geometry = _geometry_lookup(model_config)

    # Get channels - try multiple attribute names
    channels = None
    for attr_name in ["channels", "classes", "class_names"]:
        channels = _get_model_metadata(model_config, attr_name, geometry)
        if channels:
            break
    made_up = {}
    if channels is None:
        channels = made_up["channels"] = ["mito"]  # Default fallback
    channels = _normalize_metadata_list(channels, ["mito"])

    # Get voxel sizes
    input_voxel_size = _get_model_metadata(model_config, "input_voxel_size", geometry)
    if input_voxel_size is None:
        input_voxel_size = made_up["input_voxel_size"] = [16, 16, 16]
    output_voxel_size = _get_model_metadata(model_config, "output_voxel_size", geometry)
    if output_voxel_size is None:
        output_voxel_size = made_up["output_voxel_size"] = [16, 16, 16]
    _warn_made_up(model_config, made_up, warned)

    input_voxel_size = _normalize_metadata_list(input_voxel_size, [16, 16, 16])
    output_voxel_size = _normalize_metadata_list(output_voxel_size, [16, 16, 16])
    return channels, input_voxel_size, output_voxel_size


def _get_model_metadata(model_config, attr_name: str, geometry):
    """``model_config``'s ``attr_name``, else its geometry's (``geometry()``), else None."""
    # First try direct attribute access
    if hasattr(model_config, attr_name):
        value = getattr(model_config, attr_name, None)
        if value is not None:
            return value

    # Then the model's geometry. This used to read model_config.config,
    # which for a script, Hugging Face or DaCapo model builds the model in
    # the dashboard process -- weights download, torch.export, a CUDA
    # context -- just to read two voxel sizes and the channel names.
    # resolve_model_geometry asks the model's running server, then its
    # cache, and builds the model only when neither can answer.
    config = geometry()
    if config is not None:
        value = getattr(config, attr_name, None)
        if value is not None:
            return value

    return None


def _geometry_lookup(model_config):
    """The model's geometry (see models/geometry_cache), as a function that
    looks it up on its first call only, and never when the config says it all."""

    @functools.cache
    def geometry():
        from cellmap_flow.models.geometry_cache import resolve_model_geometry

        try:
            return resolve_model_geometry(getattr(model_config, "name", None), model_config)
        except Exception as e:
            logger.debug(f"Could not resolve the geometry of {model_config}: {e}")
            return None

    return geometry


def _warn_made_up(model_config, made_up: dict, warned: set) -> None:
    """Say, once per model, which of its settings the trainer is guessing.

    A model that says nothing of its channels or voxel sizes is trained as
    if it predicted mito at 16 nm, which is quietly wrong for most models.
    """
    name = getattr(model_config, "name", None)
    if not made_up or name in warned:
        return
    warned.add(name)
    guesses = ", ".join(f"{key}={value}" for key, value in made_up.items())
    logger.warning(
        f"Model {name!r} does not say its {', '.join(made_up)}; training it "
        f"with {guesses}. Set them in the model's config if that is wrong."
    )


def _normalize_metadata_list(value, default):
    """Return model metadata as a plain list for CLI serialization."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    return list(value)


def extract_data_path_from_corrections(corrections_path: Path) -> str:
    """Extract dataset path from corrections metadata.

    The manifest's raw_dataset_path first -- it is what the trainer reads
    -- and only then the first correction zarr's attrs, which a crop zarr
    may not have.
    """
    from cellmap_flow.finetune.session.manifest import read_manifest

    try:
        raw = (read_manifest(str(corrections_path)) or {}).get("raw_dataset_path")
    except (OSError, ValueError):
        raw = None
    if raw:
        return raw

    # Look for first .zarr directory
    zarr_dirs = sorted(corrections_path.glob("*.zarr"))
    if not zarr_dirs:
        raise ValueError("No .zarr directories found in corrections")

    # Read .zattrs
    zattrs_file = zarr_dirs[0] / ".zattrs"
    if not zattrs_file.exists():
        raise ValueError("No .zattrs metadata found in corrections")

    with open(zattrs_file) as f:
        metadata = json.load(f)

    if "dataset_path" not in metadata:
        raise ValueError("No 'dataset_path' found in corrections metadata")

    return metadata["dataset_path"]


# Values in this command survive two rounds of shell quoting: the one bsub
# starts on the exec host, and LSF's own handling, which re-wraps the whole
# `bash -c` argument in single quotes. A single quote of ours therefore closes
# LSF's and the argument word-splits. That is not hypothetical:
#
#     --offsets '[[1, 0, 0], [0, 1, 0], [0, 0, 1]]'
#
# reached the trainer as the bare word "[[1," with the rest scattered as stray
# arguments, and json.loads died with "Expecting value: line 1 column 5".
# Double quotes nest inside LSF's single quotes safely, so quote with those.
_SHELL_SAFE = frozenset(string.ascii_letters + string.digits + "@%+=:,./-_")


def _sh_quote(part: str) -> str:
    """Shell-quote without ever emitting a single quote."""
    part = str(part)
    if part and all(c in _SHELL_SAFE for c in part):
        return part
    escaped = (
        part.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    return f'"{escaped}"'


def build_command(
    *,
    model_config,
    model_type: str,
    checkpoint_path: Optional[Path],
    corrections_path: Path,
    output_dir: Path,
    log_file: Path,
    channels: List[str],
    input_voxel_size: List[int],
    output_voxel_size: List[int],
    lora_r: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    loss_type: str,
    label_smoothing: float,
    distillation_lambda: Optional[float],
    distillation_scope: str,
    margin: float,
    auto_serve: bool,
    serve_data_path: Optional[str],
    mask_unannotated: bool,
    balance_classes: bool,
    augment: bool,
    output_type: str,
    select_channel: Optional[int],
    offsets: Optional[str],
    models_dir: Optional[Path] = None,
    queue: Optional[str] = None,
    charge_group: Optional[str] = None,
) -> str:
    """Build the shell command used to launch finetuning."""
    command_parts = [
        sys.executable,
        "-m",
        "cellmap_flow.finetune.finetune_cli",
        "--model-type", model_type,
    ]

    if model_type in MODEL_ENTRY_TYPES:
        # No dedicated flags: hand the trainer the model's own entry.
        # encode_to_str() is URL-safe base64, so it needs no quoting.
        from cellmap_flow.serving.protocol import encode_to_str

        command_parts += ["--model-entry", encode_to_str(model_config.to_dict())]
    elif model_type == "huggingface":
        command_parts += ["--repo", str(model_config.repo)]
        if getattr(model_config, "revision", None):
            command_parts += ["--revision", str(model_config.revision)]
    elif checkpoint_path:
        command_parts += ["--model-checkpoint", str(checkpoint_path)]
    elif hasattr(model_config, "script_path"):
        command_parts += ["--model-script", str(model_config.script_path)]

    command_parts += [
        "--corrections", str(corrections_path),
        "--output-dir", str(output_dir),
        "--model-name", str(model_config.name),
        "--channels", *map(str, channels),
        "--input-voxel-size", *map(str, input_voxel_size),
        "--output-voxel-size", *map(str, output_voxel_size),
        "--lora-r", str(lora_r),
        "--lora-alpha", str(lora_r * 2),
        "--num-epochs", str(num_epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--loss-type", str(loss_type),
    ]

    if label_smoothing > 0:
        command_parts += ["--label-smoothing", str(label_smoothing)]
    # Passed whenever it was chosen, 0 included: leaving the flag out means
    # "unset", which the trainer turns into 1.0 when good regions exist,
    # so omitting it for 0 made "0 (Disabled)" impossible to express.
    if distillation_lambda is not None:
        command_parts += ["--distillation-lambda", str(distillation_lambda)]
    if distillation_scope == "all" and (distillation_lambda is None or distillation_lambda > 0):
        command_parts.append("--distillation-all-voxels")
    if loss_type == "margin":
        command_parts += ["--margin", str(margin)]
    if auto_serve and serve_data_path:
        command_parts += ["--auto-serve", "--serve-data-path", str(serve_data_path)]
    if mask_unannotated:
        command_parts.append("--mask-unannotated")
    if balance_classes:
        command_parts.append("--balance-classes")
    # Opt-out flag: only passed when augmentation is disabled.
    if not augment:
        command_parts.append("--no-augment")
    if output_type != "binary":
        command_parts += ["--output-type", str(output_type)]
    if select_channel is not None:
        command_parts += ["--select-channel", str(select_channel)]
    if offsets is not None:
        command_parts += ["--offsets", str(offsets)]
    if models_dir is not None:
        command_parts += ["--models-dir", str(models_dir)]
    # Only written into the serving YAMLs, so a model served from one runs
    # where this job did rather than on hard-coded defaults.
    if queue:
        command_parts += ["--queue", str(queue)]
    if charge_group:
        command_parts += ["--charge-group", str(charge_group)]

    command = " ".join(_sh_quote(part) for part in command_parts)

    # Put this interpreter's own lib directory first on the loader path.
    #
    # We launch sys.executable directly rather than through the
    # environment's activation script, so nothing sets LD_LIBRARY_PATH for
    # us -- and LSF runs the exec host's login shell first, which can put
    # system paths ahead of ours. When that happens the system
    # libstdc++.so.6 is loaded before anything from this environment, and
    # the first extension module built against a newer toolchain fails
    # with "version `CXXABI_1.3.15' not found" even though the
    # environment ships a libstdc++ that has it. Seen with scipy pulled in
    # via cellpose on a GCC 13+ build.
    env_lib = os.path.join(sys.prefix, "lib")
    loader_path = (
        f"LD_LIBRARY_PATH={_sh_quote(env_lib)}"
        '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} '
    )
    # stdbuf on *both* sides. The trainer already flushes every line it
    # prints, but tee writes to the log file through stdio, which is
    # block-buffered when the destination is not a terminal -- so roughly
    # 8KB of output, five to ten epochs' worth, landed in the file at
    # once and the dashboard showed nothing in between.
    return (
        f"{loader_path}stdbuf -oL {command} 2>&1 "
        f"| stdbuf -oL tee {_sh_quote(log_file)}"
    )


def launch(job_name: str, command: str, queue: Optional[str], charge_group: Optional[str]):
    """Start ``command`` (build_command's) as the job ``job_name``.

    On LSF with one GPU when bsub is available, for the dashboard's walltime
    (``g.walltime``) or else the site's; otherwise as a process here.
    Returns its LSFJob or LocalJob; raises RuntimeError if it did not start.
    """
    # Check if bsub is available
    if is_bsub_available():
        logger.info("Submitting to LSF cluster via bsub")
        try:
            # Training runs epochs, not chunks, so it is the likeliest
            # thing here to outlive the queue's 120-minute default.
            from cellmap_flow.globals import g as _g

            lsf_job = jobs_lsf.submit(JobSpec(
                name=job_name,
                # A shell line: it sets LD_LIBRARY_PATH and pipes through tee.
                shell=command,
                queue=queue,
                charge_group=charge_group,
                gpus=1,
                cpus=4,
                walltime=getattr(_g, "walltime", None) or current_site().default_walltime,
            ))
            logger.info(f"Submitted LSF job {lsf_job.job_id} for finetuning")
        except Exception as e:
            logger.error(f"Failed to submit job to LSF: {e}")
            raise RuntimeError(f"Job submission to LSF failed: {e}")
    else:
        # Fallback to local execution
        logger.info("bsub not available - running finetuning locally")
        try:
            # command is a shell command: it sets LD_LIBRARY_PATH as a
            # prefix assignment and pipes through tee. run_locally splits
            # with shlex and runs shell=False on purpose, so handing it
            # this string would exec "LD_LIBRARY_PATH=..." as a program.
            # Give it an argv list with an explicit shell instead -- the
            # list form skips run_locally's shlex.split entirely.
            lsf_job = run_locally(
                command=["bash", "-c", command],
                name=job_name,
                # The command tees its own output to training_log.txt;
                # a second copy under ~/.cellmap_flow/server_logs is noise.
                log_file=os.devnull,
            )
            logger.info(f"Started local finetuning job (PID: {lsf_job.process.pid})")
        except Exception as e:
            logger.error(f"Failed to start local job: {e}")
            raise RuntimeError(f"Local job execution failed: {e}")
    return lsf_job
