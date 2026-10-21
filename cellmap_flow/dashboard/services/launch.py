"""Start the models picked on the Models tab, and show each in the viewer.

- ``update_run_models()``: stop and forget the models no longer picked,
  start the ones newly picked, each in its own thread. A finetune job's
  server is the Finetune tab's to stop, never this one's.
- ``run_model()`` / ``run_hf_model()`` / ``run_bioimage_model()`` /
  ``run_cellpose_model()``: start one catalog, Hugging Face, BioImage Model
  Zoo or Cellpose model's inference server (``start_hosts``, which records
  the job) and add its layer, the one Submit would give it. Each is served
  from its type's ``default_env``, when the type sets one
  (``serving.launch.server_argv_for``): a zoo model from pixi's
  ``bioimageio`` environment, a Cellpose one from ``cellpose4``.
- ``loaded_models()`` / ``remove_model()``: the Models tab's *Loaded
  models* list, every model running or starting with where it came from,
  and its ✕: stop that one model and take its layer away, whatever else is
  ticked on the tab and not submitted.
"""

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.serving.launch import server_command, server_command_for
from cellmap_flow.jobs.launch import start_hosts, starting_jobs
from cellmap_flow.jobs.spec import JobStartError
from cellmap_flow.viewer.raw import PREDICTION_COLORS
from cellmap_flow.models import bioimage_catalog
from cellmap_flow.models.models_config import (
    BioModelConfig, CellposeModelConfig, FinetuneModelConfig, HuggingFaceModelConfig,
)
from cellmap_flow.viewer.layers import prediction_layer, replace_layer
import threading
from typing import List
import re
import functools
import logging

logger = logging.getLogger(__name__)


def _sanitize_job_name(name: str) -> str:
    """A Hugging Face model's name: its repo's last part, spaces and hyphens
    made underscores. It names the bsub job, the viewer layer and the
    model's config. (routes/blockwise has a _sanitize_job_name of its own,
    which keeps hyphens in a task's name: making the two one would rename
    either the Hugging Face models or the blockwise tasks.)"""
    return re.sub(r"[\s\-]+", "_", name)


def _start(command, name):
    """start_hosts(), or None after logging why the job did not start.

    These run in the dashboard's launch threads: an uncaught JobStartError
    (including a bsub timeout) only reached stderr, never the log panel.

    A model removed from the Loaded models list while it was starting
    (``remove_model``) is not left running: its job is killed and
    forgotten as it comes up, and its failure to start, when the removal
    killed it in its queue, is no error.
    """
    session = get_session()
    try:
        job = start_hosts(
            command, job_name=name, queue=session.queue, charge_group=session.charge_group
        )
    except JobStartError as e:
        if _withdrawn():
            logger.info(f"Stopped '{name}' while it was starting: it was removed from the loaded models")
            return None
        logger.error(f"Could not start model '{name}': {e}")
        return None
    if _withdrawn():
        logger.info(f"Stopping '{name}', which was removed from the loaded models while it was starting")
        # No layer of its own yet: one of its name is a later launch's.
        _stop_jobs([job], drop_layers=False)
        return None
    return job


def _show(job, st_data):
    """Add the started model's layer, the one Submit would give it; none for
    a model removed from the Loaded models list since its job came up
    (``remove_model`` stopped that job)."""
    if _withdrawn():
        return
    session = get_session()
    names = [j.model_name for j in session.jobs]
    index = names.index(job.model_name) if job.model_name in names else 0
    layer = prediction_layer(
        job.model_name, job.host, st_data, dataset_path=session.dataset_path, postprocess=session.postprocess,
        shader=session.shaders.get(job.model_name), shader_controls=session.shader_controls.get(job.model_name),
        color=PREDICTION_COLORS[index % len(PREDICTION_COLORS)],
    )
    with session.viewer.txn() as s:
        # A model started again keeps what the user set on its layer.
        replace_layer(s.layers, job.model_name, layer)


def _reported(launch):
    """Log what a launch thread raises: it runs on its own thread, whose
    exceptions only reach the terminal, so a model that failed before its
    job was submitted looked, on the page, like one that was never started."""
    @functools.wraps(launch)
    def run(*args, **kwargs):
        try:
            return launch(*args, **kwargs)
        except Exception as e:
            logger.error(f"Could not start a model ({launch.__name__}): {e}", exc_info=True)
    return run


@_reported
def run_model(model_path, name, st_data):
    if model_path is None or model_path == "":
        logger.error(f"Model path is empty for {name}")
        return
    session = get_session()
    command = server_command_for("cellmap", {"folder_path": model_path, "name": name}, session.dataset_path, session.resample)
    logger.info(f"To be submitted command : {command}")
    job = _start(command, name)
    if job is not None:
        _show(job, st_data)


@_reported
def run_hf_model(repo, name, st_data):
    """Run a Hugging Face model by repo ID."""
    name = _sanitize_job_name(name)
    session = get_session()
    command = server_command_for("huggingface", {"repo": repo, "name": name}, session.dataset_path, session.resample)
    logger.info(f"To be submitted HF command : {command}")
    job = _start(command, name)
    if job is not None:
        _show(job, st_data)


@_reported
def run_model_config(model_config, st_data):
    """Run a model given as its config (an entry the Models tab's Add built),
    in its environment, as Submit runs the catalog's."""
    session = get_session()
    command = server_command(model_config, session.dataset_path, session.resample)
    logger.info(f"To be submitted command : {command}")
    job = _start(command, model_config.name)
    if job is not None:
        _show(job, st_data)


def bioimage_job_name(key: str) -> str:
    """A zoo model's job, layer and config name, from the key it is loaded
    by: "affable-shark" is affable_shark. A DOI's slash and dots go too."""
    return re.sub(r"\W+", "_", key).strip("_")


@_reported
def run_bioimage_model(params, st_data):
    """Run a BioImage Model Zoo model: ``params`` are its BioModelConfig
    arguments, name included (``bioimage_catalog.bioimage_entry``)."""
    session = get_session()
    command = server_command_for("bioimage", params, session.dataset_path, session.resample)
    logger.info(f"To be submitted bioimage command : {command}")
    job = _start(command, params["name"])
    if job is not None:
        _show(job, st_data)


def _bioimage_params(selections):
    """bioimage_entry()'s arguments for each ticked zoo model ({"id", "voxel_size"}).

    Built before anything is stopped or started, so a model whose voxel size
    is missing refuses the whole Submit (ValueError) rather than half of it.
    The listed id may be a DOI; the model is loaded by its nickname when the
    cached list has one.
    """
    params = []
    for selection in selections:
        found = bioimage_catalog.find_bioimage_model(selection["id"])
        key = found["key"] if found else selection["id"]
        voxel_size = selection.get("voxel_size")
        trained = bioimage_catalog.trained_at(found or key) or {}
        if voxel_size is None and trained.get("voxel_size"):
            voxel_size = trained["voxel_size"]
            logger.info(f"{key}: at {voxel_size} nm, the voxel size it was trained at ({trained['trained_on']})")
        if voxel_size is None and found is not None and bioimage_job_name(key) not in running_names():
            # Its server would refuse a model with no voxel size after its job
            # started, where nothing on the page shows it: refuse here.
            try:
                declared = bioimage_catalog.declared_voxel_size(found)
            except bioimage_catalog.ZooIndexError as e:
                # Its server could not read the description either, so it
                # would fail the same way: refuse here too.
                raise ValueError(
                    f"Could not tell whether {found['name']} ({key}) says what voxel size it was "
                    f"trained at ({e}): enter one (nm) in its row"
                ) from e
            if declared is None:
                raise ValueError(
                    f"{found['name']} ({key}) does not say what voxel size it was trained at: "
                    "enter one (nm) in its row"
                )
        params.append(bioimage_catalog.bioimage_entry(key, voxel_size, bioimage_job_name(key)))
    return params


# The Cellpose models the Models tab lists: (name, what it is). Not the DINO
# ones (cpdino, cpdino-vitb): they need facebookresearch's dinov3 package,
# which pixi's cellpose4 environment does not have, so their servers would
# fail on startup. They can still be added by name or in a YAML, where an
# env can give them one that has it.
CELLPOSE_MODELS = {
    "cpsam_v2": ("Cellpose-SAM v2", "the default: Cellpose-SAM retrained, the best of Cellpose 4's models"),
    "cpsam": ("Cellpose-SAM", "the first Cellpose-SAM (Cellpose 4.0's)"),
}


def cellpose_job_name(model: str, output: str, do_3d: bool = False) -> str:
    """A Cellpose model's job, layer and config name: "cpsam_v2" serving all
    its channels (flows, the default) is cellpose_sam_v2, its probability
    alone cellpose_sam_v2_probability and its masks cellpose_sam_v2_masks;
    in 3D each ends in _3d.

    The output and 3D are in the name, so one model's outputs can run side
    by side (the probability to look at, the masks to proofread; 2D beside
    3D to compare), each its own job and layer; the default output keeps
    the plain name. The voxel size and the slice linking are not: changing
    them restarts that job instead (``update_run_models``), rather than
    leaving the old one running beside.
    """
    base = "cellpose_" + re.sub(r"^cp", "", model)
    if output != "flows":
        base += "_" + output
    if do_3d:
        base += "_3d"
    return re.sub(r"\W+", "_", base).strip("_")


def is_panel_cellpose(mc) -> bool:
    """Whether ``mc`` is one of the Cellpose panel's own rows: a model it
    lists, named as the row names its job, so its Submit names it alike."""
    return (isinstance(mc, CellposeModelConfig) and mc.pretrained_model in CELLPOSE_MODELS
            and mc.name == cellpose_job_name(mc.pretrained_model, mc.output, mc.do_3d))


@_reported
def run_cellpose_model(params, st_data):
    """Run a Cellpose model: ``params`` are its CellposeModelConfig
    arguments, name included (``_cellpose_params``)."""
    session = get_session()
    command = server_command_for("cellpose", params, session.dataset_path, session.resample)
    logger.info(f"To be submitted cellpose command : {command}")
    job = _start(command, params["name"])
    if job is not None:
        _show(job, st_data)


def _cellpose_params(selections):
    """CellposeModelConfig arguments for each ticked Cellpose model
    ({"model", "voxel_size", "output", "do_3d", "stitch_threshold"}).

    Built, and checked by building the config, before anything is stopped
    or started, so a blank voxel size or a bad setting refuses the whole
    Submit (ValueError) rather than half of it. There is no voxel size to
    fall back on, as there is for some zoo models: Cellpose sees any scale,
    and segments well only at the one where objects are about 30 voxels
    across, which only the user knows. ``stitch_threshold`` is read for
    2D masks only (the tab sends it with those only), and 0 is left out.
    """
    params, names = [], set()
    for selection in selections:
        model, output = selection["model"], selection.get("output") or "flows"
        do_3d = bool(selection.get("do_3d"))
        label = CELLPOSE_MODELS[model][0] if model in CELLPOSE_MODELS else model
        if model not in CELLPOSE_MODELS:
            raise ValueError(f"{model} is not one of the Models tab's Cellpose models ({', '.join(CELLPOSE_MODELS)})")
        if not selection.get("voxel_size"):
            raise ValueError(
                f"{label} ({model}) needs a voxel size: enter one (nm) in its row, the scale at which "
                "the objects are about 30 voxels across"
            )
        entry = {"pretrained_model": model, "voxel_size": selection["voxel_size"], "output": output,
                 "name": cellpose_job_name(model, output, do_3d)}
        if do_3d:
            entry["do_3d"] = True
        elif output == "masks" and selection.get("stitch_threshold"):
            entry["stitch_threshold"] = selection["stitch_threshold"]
        if entry["name"] in names:
            raise ValueError(f"{label} ({model}) is ticked twice with output {output}"
                             f"{' in 3D' if do_3d else ''}: untick one")
        names.add(entry["name"])
        try:
            CellposeModelConfig(**entry)
        except ValueError as e:
            raise ValueError(f"{label} ({model}): {e}") from e
        params.append(entry)
    return params


def _settings_changed(params) -> bool:
    """Whether the running Cellpose model named ``params["name"]`` was started
    with other settings (a voxel size, a slice linking) than ``params``."""
    for mc in get_session().models_config:
        if isinstance(mc, CellposeModelConfig) and mc.name == params["name"]:
            return mc.to_dict() != CellposeModelConfig(**params).to_dict()
    return False


# The names whose launch thread has not returned yet: from Submit until the
# job is in the session's jobs, which can be minutes (bsub, the queue, the
# model loading). They count as running, so a second Submit in that time
# (Resample toggled, say) does not start the same model again: two
# "impartial_shrimp" jobs once ran side by side, one resampling and one not,
# and the layer showed whichever came up last. {name: its _Launch}.
_launching: dict = {}
_launching_lock = threading.Lock()


class _Launch:
    """One launch thread's model. ``withdrawn`` once the model is removed
    from the Loaded models list before its launch returned
    (``remove_model``): ``_start`` then stops whatever it starts."""

    def __init__(self, name):
        self.name, self.withdrawn = name, False


# The launch the current thread runs, for _start to ask whether it was withdrawn.
_this_launch = threading.local()


def _withdrawn() -> bool:
    """Whether the launch this thread runs has been withdrawn (``remove_model``)."""
    launch = getattr(_this_launch, "current", None)
    return launch is not None and launch.withdrawn


def _launch(name, target, *args, daemon=False):
    """Run ``target(*args)`` in its own thread, with ``name`` counted as
    running (``running_names``) until it returns, or until it is withdrawn:
    then a Submit may start the model again, in a launch of its own."""
    launch = _Launch(name)
    with _launching_lock:
        _launching[name] = launch

    def run():
        _this_launch.current = launch
        try:
            target(*args)
        finally:
            _this_launch.current = None
            with _launching_lock:
                if _launching.get(name) is launch:
                    del _launching[name]

    thread = threading.Thread(target=run, daemon=daemon)
    thread.start()
    return thread


def running_names() -> set:
    """The models running, or on their way: the session's jobs and the
    launches not finished yet."""
    with _launching_lock:
        launching = set(_launching)
    return {job.model_name for job in get_session().jobs} | launching


def _owned_by_finetune(job) -> bool:
    """A finetune job's server: the Finetune tab's to stop, never the Models tab's."""
    return getattr(job, "owned_by_finetune", False)


def _stop_jobs(jobs, *, drop_layers=True):
    """Stop ``jobs`` and forget them (a killed job left among the jobs counts as running).

    The session's shaders for their names stay, so a model started again is
    drawn with what the user set on it; only the ✕ forgets them. Jobs are
    forgotten by identity: another of the same name may be a later launch's.
    """
    session = get_session()
    if drop_layers and jobs and session.viewer is not None:
        with session.viewer.txn() as s:
            for name in {job.model_name for job in jobs}:
                if name in s.layers:
                    del s.layers[name]
    for job in jobs:
        if not _owned_by_finetune(job):
            job.kill()
    session.jobs = [j for j in session.jobs if not any(j is job for job in jobs)]


def configs_by_name() -> dict:
    return {getattr(mc, "name", None): mc for mc in get_session().models_config}


def drop_config(name):
    """Take the model config named ``name`` out of the session's models;
    nothing when none is. Export Config and the pipeline builder list what
    is left."""
    session = get_session()
    session.models_config = [mc for mc in session.models_config if getattr(mc, "name", None) != name]


def _catalog_groups() -> dict:
    return {name: group for group, models in get_session().model_catalog.items() for name in models}


def update_run_models(names: List[str], hf_repos: List[str] = None, bioimage_models: List[dict] = None,
                      cellpose_models: List[dict] = None):
    """Run ``names`` (catalog models), ``hf_repos``, ``bioimage_models``
    ({"id", "voxel_size"} each) and ``cellpose_models`` ({"model",
    "voxel_size", "output", "do_3d", "stitch_threshold"} each), and stop every other
    model. A ValueError, before anything changes, when a zoo model needs a
    voxel size or a Cellpose model's settings are missing or wrong."""
    session = get_session()
    if hf_repos is None:
        hf_repos = []
    bioimage_params = _bioimage_params(bioimage_models or [])
    cellpose_params = _cellpose_params(cellpose_models or [])

    # A running Cellpose model whose voxel size, output or slice linking was
    # changed is stopped and started again with them: its name does not say
    # them, so it would otherwise be kept as it was, and the change silently
    # ignored. Also when the same name is ticked in the model list too (a
    # YAML's "cellpose_sam" serving probability, and the Cellpose panel's
    # cpsam with flows): the panel's settings are the ones asked for.
    restarted = {p["name"] for p in cellpose_params if _settings_changed(p)}
    names = [name for name in names if name not in restarted]
    all_names = (names + [_sanitize_job_name(repo.split("/")[-1]) for repo in hf_repos]
                 + [p["name"] for p in bioimage_params])
    all_names += [p["name"] for p in cellpose_params if p["name"] not in restarted]
    # Not a finetune job's server (finetune_layers marks it): it is the
    # training job itself, and its name, new with each iteration, has no
    # box on a Models tab rendered before it, so every Submit bkilled it.
    to_be_killed = [j for j in session.jobs if j.model_name not in all_names and not _owned_by_finetune(j)]
    names_running = running_names() - {j.model_name for j in to_be_killed}

    threads = []
    st_data = session.layer_blob()

    print(f"Current catalog: {session.model_catalog}")
    _stop_jobs(to_be_killed)
    # Launch local catalog models
    for _, group in session.model_catalog.items():
        for name, model_path in group.items():
            if name in names and name not in names_running:
                logger.info(f"To be submitted model : {model_path}")
                threads.append(_launch(name, run_model, model_path, name, st_data))

    # Launch Hugging Face models
    for repo in hf_repos:
        hf_name = _sanitize_job_name(repo.split("/")[-1])
        if hf_name not in names_running:
            logger.info(f"To be submitted HF model : {repo}")
            # Create and store HuggingFaceModelConfig for pipeline builder
            hf_config = HuggingFaceModelConfig(repo=repo, name=hf_name)
            existing_names = [getattr(mc, 'name', None) for mc in session.models_config]
            if hf_name not in existing_names:
                session.models_config.append(hf_config)
            threads.append(_launch(hf_name, run_hf_model, repo, hf_name, st_data))

    # Launch BioImage Model Zoo models. Their config is kept, as a
    # Hugging Face model's is, for the pipeline builder and for the
    # Models tab to tick them on its next render (index_page). One left
    # from an earlier run is replaced: its voxel size may be another.
    for params in bioimage_params:
        if params["name"] not in names_running:
            logger.info(f"To be submitted bioimage model : {params}")
            session.models_config = [
                mc for mc in session.models_config
                if not (isinstance(mc, BioModelConfig) and mc.name == params["name"])
            ] + [BioModelConfig(**params)]
            threads.append(_launch(params["name"], run_bioimage_model, params, st_data))

    # Launch Cellpose models, keeping their config as a zoo model's is.
    for params in cellpose_params:
        if params["name"] not in names_running:
            logger.info(f"To be submitted cellpose model : {params}")
            session.models_config = [
                mc for mc in session.models_config
                if not (isinstance(mc, CellposeModelConfig) and mc.name == params["name"])
            ] + [CellposeModelConfig(**params)]
            threads.append(_launch(params["name"], run_cellpose_model, params, st_data))

    # Launch the other models ticked by name: one that Add a model
    # started (routes/model_resolve), or that the YAML the dashboard was
    # started with did, stopped since (unticked, or removed from the
    # Loaded models list) and ticked again. No catalog entry names them,
    # so ticking one again did nothing; their config starts them, as Add
    # did. Not a finetuned model's: its server is its finetune job's.
    in_catalog = _catalog_groups()
    configs = configs_by_name()
    for name in dict.fromkeys(names):
        config = configs.get(name)
        if (name in in_catalog or config is None or isinstance(config, FinetuneModelConfig)
                or name in running_names()):
            continue
        logger.info(f"To be submitted model, from its config : {name}")
        threads.append(_launch(name, run_model_config, config, st_data))
    # for thread in threads:
    #     thread.join()


# Where each model on the Loaded models list came from: the "source" of its
# entry, which models-tab.js labels. The browser's backend
# (web/src/backend/routes/models.ts) answers the same keys.
SOURCES = ("catalog", "huggingface", "bioimage", "cellpose", "added", "finetuned")


def _finetune_note(training):
    """Why a finetune job's model has no ✕ while ``training`` (its
    FinetuneJob) is alive, or what ✕ does once it has ended; and whether it
    has ended."""
    from cellmap_flow.finetune.job_manager.state import TERMINAL_STATUSES

    if training is None:
        return True, "Its finetune job is not known here any more: ✕ takes its layer away."
    status = training.status.value
    if training.status in TERMINAL_STATUSES:
        return True, f"Its finetune job has ended ({status}), and its server with it: ✕ takes its layer away."
    return False, (f"Served by its finetune job ({status}), which Restart trains again. Stop that job on "
                   "the Finetune tab (Cancel) to stop serving it; ✕ then takes its layer away.")


def _loaded_entry(name, job, config, catalog_group):
    """One model's entry of ``loaded_models``."""
    entry = {"name": name, "status": "running" if job is not None else "starting", "removable": True,
             "note": None, "hf_id": None, "zoo_id": None, "cellpose": None}
    if _owned_by_finetune(job) or isinstance(config, FinetuneModelConfig):
        from cellmap_flow.dashboard.finetune_layers import training_job_of

        training = training_job_of(job) if _owned_by_finetune(job) else None
        base = training.model_name if training is not None else None
        entry.update(source="finetuned", detail=f"from {base}" if base else None)
        if _owned_by_finetune(job):
            ended, note = _finetune_note(training)
            entry.update(removable=ended, note=note, status="running" if not ended else "ended")
    elif isinstance(config, HuggingFaceModelConfig):
        entry.update(source="huggingface", detail=config.repo, hf_id=config.repo)
    elif isinstance(config, BioModelConfig):
        zoo_id = bioimage_catalog.entry_model_id(config)
        entry.update(source="bioimage", detail=zoo_id, zoo_id=zoo_id)
    elif isinstance(config, CellposeModelConfig):
        output = config.output + (", 3D" if config.do_3d else "")
        entry.update(source="cellpose", detail=f"{config.pretrained_model}, {output}")
        # Its row in the Cellpose panel, to untick.
        if is_panel_cellpose(config):
            entry["cellpose"] = {"model": config.pretrained_model, "output": config.output, "do_3d": config.do_3d}
    elif catalog_group is not None:
        entry.update(source="catalog", detail=catalog_group)
    else:
        from cellmap_flow.models.registry import cli_name_of

        entry.update(source="added", detail=cli_name_of(type(config)) if config is not None else None)
    return entry


def loaded_models() -> list:
    """The models running and starting, for the Models tab's Loaded models
    list: the session's jobs in order, then the launches not returned yet.

    Each is ``{"name", "source"`` (one of SOURCES), ``"detail"`` (the
    catalog group, the repo, the zoo id, the Cellpose model and output, the
    model type, or the base a finetune started from), ``"status"``
    ("running", "starting", or "ended" for a finetune job's model whose job
    has), ``"removable"``, ``"note"`` (why it cannot be removed, or what ✕
    does), and what the tab's rows know it by, to untick them: ``"hf_id"``
    (the Hugging Face row's repo), ``"zoo_id"`` (the zoo row's id) and
    ``"cellpose"`` ({model, output, do_3d} of its Cellpose panel row)``}``.
    """
    session = get_session()
    configs, groups = configs_by_name(), _catalog_groups()
    jobs = list(session.jobs)
    with _launching_lock:
        launching = list(_launching)  # a withdrawn launch is no longer among them
    loaded, seen = [], set()
    for name, job in [(j.model_name, j) for j in jobs] + [(name, None) for name in launching]:
        if name in seen:
            continue
        seen.add(name)
        loaded.append(_loaded_entry(name, job, configs.get(name), groups.get(name)))
    return loaded


class NotRemovable(Exception):
    """A model whose ✕ is disabled: a finetune job's, while the job is alive."""


def remove_model(name: str) -> list:
    """Stop the one model ``name`` and take its layer away; every other
    model, and anything ticked on the Models tab and not submitted, stays as
    it is. Returns the Loaded models list as it is then.

    A running model's job is killed and forgotten, as Submit does for one
    unticked. A model still starting is withdrawn: a job of it waiting in a
    queue is killed now, and whatever its launch still starts is stopped
    as it comes up (``_start``), so a Submit may start it again at once. A
    finetune job's model is that job's server: while the job is alive it
    is the Finetune tab's to stop (NotRemovable); once the job has ended,
    its server has too, and only its layer and its entry among the jobs are
    left to take away. The model's config goes with it, so Export Config
    no longer lists a model taken off (a model unticked on Submit keeps
    its config, for the Models tab's rows and the pipeline builder).

    Raises LookupError when no model of that name is running or starting.
    """
    session = get_session()
    jobs = [j for j in session.jobs if j.model_name == name]
    with _launching_lock:
        launch = _launching.get(name)
    if not jobs and launch is None:
        raise LookupError(f"No model named {name} is running")
    for job in jobs:
        if _owned_by_finetune(job):
            from cellmap_flow.dashboard.finetune_layers import training_job_of

            ended, note = _finetune_note(training_job_of(job))
            if not ended:
                raise NotRemovable(f"{name}: {note}")

    if launch is not None:
        with _launching_lock:
            launch.withdrawn = True
            if _launching.get(name) is launch:
                del _launching[name]
        for job in starting_jobs():
            if getattr(job, "model_name", None) == name:
                logger.info(f"Killing '{name}', removed from the loaded models while it was starting")
                job.kill()
    _stop_jobs(jobs)
    session.forget_layer(name)
    drop_config(name)
    logger.info(f"Removed model {name}")
    return loaded_models()
