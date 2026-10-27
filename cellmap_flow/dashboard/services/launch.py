"""Start the models picked on the Models tab, and show each in the viewer.

- ``update_run_models()``: stop and forget the models no longer picked,
  start the ones newly picked, each in its own thread. A finetune job's
  server is the Finetune tab's to stop, never this one's. A chain with no
  input normalizers takes the ones the first model started declares.
- ``run_model()``: start one model's inference server
  (``serving.launch.launch``, which records the job) and add its layer, the
  one Submit would give it. Every model the tab picks goes through it, as a
  model config built from the entry its row describes
  (``registry.build_model``): a catalog, Hugging Face, BioImage Model Zoo or
  Cellpose model, one Add a model resolved (``routes/model_resolve``), or one
  ticked again by name. Each is served from its environment: its entry's,
  else its type's ``default_env`` (a zoo model from pixi's ``bioimageio``
  environment, a Cellpose one from ``cellpose4``).
- ``loaded_models()`` / ``remove_model()``: the Models tab's *Loaded
  models* list, every model running or starting with where it came from,
  and its ✕: stop that one model and take its layer away, whatever else is
  ticked on the tab and not submitted.
- ``launching()`` / ``start_error()``: whether a model's start is still
  going, and why its last one failed, which the Finetune tab's Use this
  model waits on (routes/finetune/rank_routes.py _served_geometry).
"""

from cellmap_flow.config.yaml import ConfigError
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.serving.launch import launch as launch_server
from cellmap_flow.jobs.launch import starting_jobs
from cellmap_flow.jobs.spec import JobStartError
from cellmap_flow.viewer.raw import PREDICTION_COLORS
from cellmap_flow.models import bioimage_catalog
from cellmap_flow.models.models_config import (
    BioModelConfig, CellposeModelConfig, FinetuneModelConfig, HuggingFaceModelConfig,
)
from cellmap_flow.models.registry import build_model
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


def _start(config):
    """``serving.launch.launch`` for ``config`` on the session's data, or None
    after logging why the job did not start.

    These run in the dashboard's launch threads: an uncaught JobStartError
    (including a bsub timeout) only reached stderr, never the log panel.

    A model removed from the Loaded models list while it was starting
    (``remove_model``) is not left running: its job is killed and
    forgotten as it comes up, and its failure to start, when the removal
    killed it in its queue, is no error.
    """
    session = get_session()
    name = config.name
    try:
        job = launch_server(
            config, session.dataset_path, queue=session.queue, charge_group=session.charge_group,
            resample=session.resample,
        )
    except JobStartError as e:
        if _withdrawn():
            logger.info(f"Stopped '{name}' while it was starting: it was removed from the loaded models")
            return None
        logger.error(f"Could not start model '{name}': {e}")
        with _launching_lock:
            _start_errors[name] = str(e)
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
        color=PREDICTION_COLORS[index % len(PREDICTION_COLORS)], channel=session.shown_channel(job.model_name),
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
def run_model(config, st_data):
    """Start ``config``'s inference server, in its environment, and add its
    layer with ``st_data``, the session's layer blob. The one launcher behind
    Submit, Add a model and the Training tab's Use this model."""
    job = _start(config)
    if job is not None:
        _show(job, st_data)


def _built(entry: dict, label: str):
    """``registry.build_model`` of ``entry``, a problem said as a ValueError naming ``label``."""
    try:
        return build_model(entry)
    except ConfigError as e:
        raise ValueError(f"{label}: {e}") from e


def bioimage_job_name(key: str) -> str:
    """A zoo model's job, layer and config name, from the key it is loaded
    by: "affable-shark" is affable_shark. A DOI's slash and dots go too."""
    return re.sub(r"\W+", "_", key).strip("_")


def bioimage_model_entry(model, voxel_size=None, *, online: bool = True):
    """``(entry, choice, refusal)``: the model entry the Models tab's Submit
    starts for a zoo model at ``voxel_size`` (what its row says; None when
    blank), how its voxel size was decided, and why its server would refuse
    it, or None.

    ``model`` is its row of the cached list, or the id a row was ticked by,
    which is looked up in it: the listed id may be a DOI, and the model is
    loaded by its nickname when the list has one, and named
    ``bioimage_job_name`` of that. Its voxel size is decided as its server
    decides it (``bioimage_catalog.entry_voxel_size``, ``choice``); a model
    that declares none, and that cellmap-flow does not know the training
    voxel size of, would be refused by its server after its job started,
    where nothing on the page shows it, so ``refusal`` says so in a sentence
    (unless it is running already), and the caller refuses it. ``online``
    False decides from the cached list alone, as the Finetune tab's ranking
    lists every zoo model (routes/finetune/rank_routes.py) without a fetch
    for each.
    """
    found = model if isinstance(model, dict) else bioimage_catalog.find_bioimage_model(model)
    key = found["key"] if found else model
    name = bioimage_job_name(key)
    choice = bioimage_catalog.entry_voxel_size(key, voxel_size, listed=found, online=online)
    refusal = None
    if choice.source in ("none", "unread") and name not in running_names():
        how = "does not say" if choice.source == "none" else "could not be read, so it is not known whether it says"
        refusal = f"{found['name']} ({key}) {how} what voxel size it was trained at"
    return bioimage_catalog.bioimage_entry(key, choice.voxel_size, name), choice, refusal


def _bioimage_configs(selections) -> list:
    """A BioModelConfig for each ticked zoo model ({"id", "voxel_size"}), of
    the entry ``bioimage_model_entry`` gives it.

    Built before anything is stopped or started, so a model whose voxel size
    is missing refuses the whole Submit (ValueError) rather than half of it.
    """
    configs = []
    for selection in selections:
        entry, choice, refusal = bioimage_model_entry(selection["id"], selection.get("voxel_size"))
        if refusal:
            raise ValueError(f"{refusal}: enter one (nm) in its row")
        if choice.source == "trained":
            logger.info(f"{entry['model']}: at {choice.voxel_size} nm, the voxel size it was trained at "
                        f"({choice.trained['trained_on']})")
        configs.append(_built(entry, entry["model"]))
    return configs


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


def _cellpose_configs(selections) -> list:
    """A CellposeModelConfig for each ticked Cellpose model
    ({"model", "voxel_size", "output", "do_3d", "stitch_threshold"}).

    Built before anything is stopped or started, so a blank voxel size or a
    bad setting refuses the whole Submit (ValueError) rather than half of
    it. There is no voxel size to fall back on, as there is for some zoo
    models: Cellpose sees any scale, and segments well only at the one
    where objects are about 30 voxels across, which only the user knows.
    ``stitch_threshold`` is read for 2D masks only (the tab sends it with
    those only), and 0 is left out.
    """
    configs, names = [], set()
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
        entry = {"type": "cellpose", "pretrained_model": model, "voxel_size": selection["voxel_size"],
                 "output": output, "name": cellpose_job_name(model, output, do_3d)}
        if do_3d:
            entry["do_3d"] = True
        elif output == "masks" and selection.get("stitch_threshold"):
            entry["stitch_threshold"] = selection["stitch_threshold"]
        if entry["name"] in names:
            raise ValueError(f"{label} ({model}) is ticked twice with output {output}"
                             f"{' in 3D' if do_3d else ''}: untick one")
        names.add(entry["name"])
        configs.append(_built(entry, f"{label} ({model})"))
    return configs


def _settings_changed(config) -> bool:
    """Whether the running Cellpose model named as ``config`` was started
    with other settings (a voxel size, a slice linking) than ``config``'s."""
    for mc in get_session().models_config:
        if isinstance(mc, CellposeModelConfig) and mc.name == config.name:
            return mc.to_dict() != config.to_dict()
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

# Why each model's last start failed (its JobStartError's message: no queue
# took the job, it died loading), by name, until it is started again: the
# Finetune tab's Use this model, which waits for the start, says why it
# gave up (start_error).
_start_errors: dict = {}


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
        _start_errors.pop(name, None)

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


def launching(name) -> bool:
    """Whether ``name``'s start is still going: its launch thread waits for
    its job's host (jobs/launch.py start_hosts: the job pending on a GPU
    queue, falling back to the next, or its server loading the model)."""
    with _launching_lock:
        return name in _launching


def start_error(name):
    """Why ``name``'s last start failed, or None: it has not failed, or was
    started again since."""
    with _launching_lock:
        return _start_errors.get(name)


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


def _keep_config(config):
    """Put ``config`` among the session's models, in place of one of its
    name: for the pipeline builder, Export Config and the Models tab's rows
    on its next render (index_page). One left from an earlier run is
    replaced, its settings (a zoo model's voxel size) may be others."""
    session = get_session()
    models = list(session.models_config)
    if any(mc is config for mc in models):
        return
    if any(getattr(mc, "name", None) == config.name for mc in models):
        models = [config if getattr(mc, "name", None) == config.name else mc for mc in models]
    else:
        models.append(config)
    session.models_config = models


def _catalog_groups() -> dict:
    return {name: group for group, models in get_session().model_catalog.items() for name in models}


def _picked(names, hf_repos, bioimage_models, cellpose_models) -> list:
    """``[(name, config)]`` for everything ticked, in the order the tab lists
    them: the catalog's models, Hugging Face's, the zoo's, the Cellpose
    panel's, then the models ticked by name alone (one that Add a model
    started, or that the YAML the dashboard was started with did, stopped
    since and ticked again: no catalog entry names them, so their config
    starts them; not a finetuned model's, whose server is its finetune
    job's).

    Each config is built here, before anything is stopped or started, so a
    bad entry refuses the whole Submit (ValueError) rather than half of it.
    A name picked twice keeps its first config: the Cellpose panel's row
    wins over a YAML's model of the same name (a YAML's "cellpose_sam"
    serving probability, and the panel's cpsam with flows), the panel's
    settings being the ones asked for.
    """
    session = get_session()
    picked, taken = [], set()

    def add(name, config):
        if name not in taken:
            taken.add(name)
            picked.append((name, config))

    for group in session.model_catalog.values():
        for name, folder in group.items():
            if name in names:
                add(name, _built({"type": "cellmap", "folder_path": folder, "name": name}, name))
    for repo in hf_repos:
        name = _sanitize_job_name(repo.split("/")[-1])
        add(name, _built({"type": "huggingface", "repo": repo, "name": name}, repo))
    for config in _bioimage_configs(bioimage_models) + _cellpose_configs(cellpose_models):
        add(config.name, config)
    configs = configs_by_name()
    for name in names:
        config = configs.get(name)
        if config is not None and not isinstance(config, FinetuneModelConfig):
            add(name, config)
    return picked


def update_run_models(names: List[str], hf_repos: List[str] = None, bioimage_models: List[dict] = None,
                      cellpose_models: List[dict] = None):
    """Run ``names`` (catalog models, and models ticked by name), ``hf_repos``,
    ``bioimage_models`` ({"id", "voxel_size"} each) and ``cellpose_models``
    ({"model", "voxel_size", "output", "do_3d", "stitch_threshold"} each),
    and stop every other model. A ValueError, before anything changes, when
    an entry cannot be built: a zoo model needs a voxel size, a Cellpose
    model's settings are missing or wrong, a catalog folder is not there.

    When the chain has no input normalizers, the first model started that
    declares some (an onnx model's manifest) sets them, as part of its start
    (routes.pipeline.set_declared_input_norm); any other chain is left as it
    is, and only the page's Apply changes it."""
    session = get_session()
    picked = _picked(names, hf_repos or [], bioimage_models or [], cellpose_models or [])

    # A running Cellpose model whose voxel size, output or slice linking was
    # changed is stopped and started again with them: its name does not say
    # them, so it would otherwise be kept as it was, and the change silently
    # ignored.
    restarted = {name for name, config in picked
                 if isinstance(config, CellposeModelConfig) and _settings_changed(config)}
    kept = (set(names) | {name for name, _ in picked}) - restarted
    # Not a finetune job's server (finetune_layers marks it): it is the
    # training job itself, and its name, new with each iteration, has no
    # box on a Models tab rendered before it, so every Submit bkilled it.
    to_be_killed = [j for j in session.jobs if j.model_name not in kept and not _owned_by_finetune(j)]
    names_running = running_names() - {j.model_name for j in to_be_killed}
    starting = [(name, config) for name, config in picked if name not in names_running]

    _stop_jobs(to_be_killed)
    # A chain with no input normalizers takes the first starting model's
    # declared ones, before the blob its layers are drawn with is taken, so
    # they are drawn through them (routes.pipeline.set_declared_input_norm).
    # After the stop, so the layers of the models stopped are not redrawn.
    from cellmap_flow.dashboard.routes.pipeline import set_declared_input_norm

    set_declared_input_norm([config for _, config in starting])
    st_data = session.layer_blob()
    for name, config in starting:
        _keep_config(config)
        # launch_entry, not to_dict(): a Hugging Face model's to_dict() fetches its metadata.
        logger.info(f"To be submitted: {name} ({config.launch_entry})")
        _launch(name, run_model, config, st_data)


# Where each model on the Loaded models list came from: the "source" of its
# entry, which js/dashboard/models/loaded.js labels. The browser's backend
# (web/src/backend/routes/models.ts) answers the same keys.
SOURCES = ("catalog", "huggingface", "bioimage", "cellpose", "added", "finetuned")


def _finetune_note(training):
    """What ✕ does to a finetune job's model, whose server is ``training``
    (its FinetuneJob): whether that job has ended, the entry's note, and,
    while the job is alive, what the ✕ asks before it cancels the job."""
    from cellmap_flow.finetune.job_manager.state import TERMINAL_STATUSES

    if training is None:
        return True, "Its finetune job is not known here any more: ✕ takes its layer away.", None
    status = training.status.value
    if training.status in TERMINAL_STATUSES:
        return True, f"Its finetune job has ended ({status}), and its server with it: ✕ takes its layer away.", None
    # Cancelling deletes nothing: each iteration exported into the run's
    # iterations/ (run_outputs), and its serving YAML is in models/.
    return False, (
        f"Served by its finetune job ({status}): ✕ cancels that job, as Cancel on the Finetune tab "
        "does, and takes its layer away."
    ), (
        f"This model is served by its finetune job ({status}). Removing it cancels that job: "
        "Restart training is no longer possible, and any training under way stops. The weights "
        f"of each iteration it finished stay saved in {training.output_dir}.\n\n"
        "Remove it and cancel the job?"
    )


def _loaded_entry(name, job, config, catalog_group):
    """One model's entry of ``loaded_models``."""
    entry = {"name": name, "status": "running" if job is not None else "starting", "removable": True,
             "note": None, "confirm": None, "hf_id": None, "zoo_id": None, "cellpose": None}
    if _owned_by_finetune(job) or isinstance(config, FinetuneModelConfig):
        from cellmap_flow.dashboard.finetune_layers import training_job_of

        training = training_job_of(job) if _owned_by_finetune(job) else None
        base = training.model_name if training is not None else None
        entry.update(source="finetuned", detail=f"from {base}" if base else None)
        if _owned_by_finetune(job):
            ended, note, confirm = _finetune_note(training)
            entry.update(note=note, confirm=confirm, status="running" if not ended else "ended")
    elif isinstance(config, HuggingFaceModelConfig):
        entry.update(source="huggingface", detail=config.repo, hf_id=config.repo)
    elif isinstance(config, BioModelConfig):
        entry.update(source="bioimage", detail=config.model, zoo_id=config.model)
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
    has), ``"removable"`` (true: every model's ✕ works), ``"note"`` (what ✕
    does, for a finetune job's), ``"confirm"`` (what ✕ asks first, when it
    does more than stop the model: a finetune job's, which it cancels; else
    None), and what the tab's rows know it by, to untick them: ``"hf_id"``
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
    """A model ✕ could not remove: a finetune job's, whose job could not be cancelled."""


def remove_model(name: str) -> list:
    """Stop the one model ``name`` and take its layer away; every other
    model, and anything ticked on the Models tab and not submitted, stays as
    it is. Returns the Loaded models list as it is then.

    A running model's job is killed and forgotten, as Submit does for one
    unticked. A model still starting is withdrawn: a job of it waiting in a
    queue is killed now, and whatever its launch still starts is stopped
    as it comes up (``_start``), so a Submit may start it again at once. A
    finetune job's model is that job's server: while the job is alive, ✕
    (once the page has asked, see _finetune_note) cancels it as the
    Finetune tab's Cancel does, and NotRemovable if it cannot; once the job
    has ended, its server has too, and only its layer and its entry among
    the jobs are left to take away. The model's config goes with it, so
    Export Config no longer lists a model taken off (a model unticked on
    Submit keeps its config, for the Models tab's rows and the pipeline
    builder).

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
            from cellmap_flow.finetune.job_manager.state import TERMINAL_STATUSES

            training = training_job_of(job)
            ended, _, _ = _finetune_note(training)
            if not ended:
                # Its server is the job itself: cancelled as the Finetune tab's Cancel does it.
                session.finetune_job_manager.cancel_job(training.job_id)
                if training.status not in TERMINAL_STATUSES:
                    raise NotRemovable(
                        f"{name}: its finetune job could not be cancelled (the dashboard's log says why); "
                        f"it is still {training.status.value}."
                    )

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
