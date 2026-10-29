"""The models to annotate for, a new annotation volume, and the user's settings.

Routes: GET ``/api/finetune/models``, POST ``/api/finetune/create-volume``,
and GET and POST ``/api/finetune/user-prefs``.

A model's entry says, with its geometry, whether it trains one channel at a
time (``channel_choice``, common.channel_choice): training's submit refuses
such a model without the channel the paint is for (``channel_pick_refusal``).

A volume is made for a model, on the grid its prediction is served on, or
before any model (a "paint-first" volume, ``make_paint_volume``) on the
raw's grid at a voxel size of the user's, the raw level the viewer shows by
default: the models answer lists the raw's levels for New Volume to offer.
A paint-first volume is not trained on (``paint_first_refusal``, which
training's submit asks): Which model to start from's Use this model carries
its paint onto a model's grid (rank_routes, through ``carry_onto_model``).
So does training's submit for a model whose grid is not the session
volume's (``grid_change`` says whether it is, and refuses a model whose
grid is not known yet).
"""

import functools
import logging
import os

from flask import jsonify, request

from cellmap_flow.dashboard.finetune_utils import (
    ensure_minio_serving,
    retire_annotation_volume,
    sync_annotation_volume_from_minio,
)
from cellmap_flow.dashboard.requests import CreateVolume, parse
from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.routes.finetune.common import (
    annotation_dtype_for,
    channel_choice,
    current_chain,
    ensure_corrections_storage,
    find_model_config,
    load_user_prefs,
    rewrite_minio_url_for_proxy,
    save_user_prefs,
    session_store,
    trains_one_channel,
    write_volume_manifest,
)
from cellmap_flow.dashboard.routes.finetune.overlay import add_annotation_layer, refresh_annotated_regions_layer
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session.volume import (
    carry_paint,
    create_volume_zarr,
    new_volume_id,
    plan_paint_volume,
    plan_volume,
    raw_voxel_sizes,
)
from cellmap_flow.models.geometry_cache import known_geometry, resolve_model_geometry
from cellmap_flow.portable import volume as portable_volume
from cellmap_flow.serving.client import (
    fetch_model_info,
    model_geometry,
    output_channel_names,
    running_job_host,
)

logger = logging.getLogger(__name__)


def serve_new_volume(geometry, corrections_dir, dataset_path, model_name, annotation_dtype="uint8"):
    """Write a new volume with ``geometry`` into ``corrections_dir`` and serve it.

    The volume records the dashboard's current normalization and
    postprocessing chains, so the trainer can reproduce the inference-side
    normalization and the finetuned model's YAML the postprocessing. Returns
    ``(volume_id, zarr_path, url)``, the URL as the browser reaches MinIO.
    Registering the volume is the caller's.
    """
    volume_id = new_volume_id()
    zarr_path = os.path.join(corrections_dir, f"{volume_id}.zarr")
    input_norm, postprocess = current_chain()
    create_volume_zarr(
        zarr_path,
        geometry,
        dataset_path=dataset_path,
        model_name=model_name,
        input_norm=input_norm,
        postprocess=postprocess,
        annotation_dtype=annotation_dtype,
    )
    minio_url = ensure_minio_serving(zarr_path, volume_id, output_base_dir=corrections_dir)
    return volume_id, zarr_path, rewrite_minio_url_for_proxy(minio_url)


def make_volume(model_name, model_config, config, corrections_dir):
    """A new annotation volume for ``model_name`` (``config`` its geometry,
    resolve_model_geometry's), over the session's dataset in
    ``corrections_dir``: written, served, registered as the session's and
    in its manifest. Returns ``(volume_id, zarr_path, minio_url, geometry)``.
    Create Volume makes one; so does Which model to start from's Use this
    model (rank_routes)."""
    session = get_session()
    geometry = plan_volume(session.dataset_path, config, resample=session.resample)
    return _made(geometry, model_name, annotation_dtype_for(model_config), corrections_dir)


def make_paint_volume(voxel_size, corrections_dir):
    """A new volume painted before any model (no ``model_name``), over the
    session's dataset in ``corrections_dir``, on the raw's grid at
    ``voxel_size`` (nm, z, y, x; None: the level the viewer shows,
    ``shown_voxel_size``): written, served and registered as the session's,
    as make_volume's, and returned as it returns one. It gets no training
    manifest, as it is not trained on (``paint_first_refusal``), and the
    session's manifest is removed (common.write_volume_manifest), so that
    it, not a volume before it, is the session's."""
    session = get_session()
    size = list(voxel_size) if voxel_size is not None else shown_voxel_size()
    geometry = plan_paint_volume(session.dataset_path, size, resample=session.resample)
    return _made(geometry, None, portable_volume.annotation_dtype(None), corrections_dir)


def _made(geometry, model_name, annotation_dtype, corrections_dir):
    """make_volume's and make_paint_volume's volume of ``geometry``, for
    ``model_name`` (None: for no model), written, served and registered as
    the session's; a model's also in the session's training manifest, which
    a volume for no model removes."""
    session = get_session()
    volume_id, zarr_path, minio_url = serve_new_volume(
        geometry, corrections_dir, session.dataset_path, model_name, annotation_dtype
    )
    session_store().register_volume(
        volume_id,
        **geometry.record(
            zarr_path,
            dataset_path=session.dataset_path,
            model_name=model_name,
            corrections_dir=corrections_dir,
        ),
    )
    # The trainer finds the volume only through this manifest, and every
    # route the session's volume (session_store's session_volume): a volume
    # for no model gets none, and takes away one naming another volume.
    write_volume_manifest(session.annotation_volumes[volume_id])
    refresh_annotated_regions_layer()
    return volume_id, zarr_path, minio_url, geometry


@functools.lru_cache(maxsize=8)
def _cached_raw_voxel_sizes(dataset_path):
    """raw_voxel_sizes of ``dataset_path``, read once (a failure is not kept, so it is asked again)."""
    return tuple(tuple(size) for size in raw_voxel_sizes(dataset_path))


def session_raw_voxel_sizes():
    """The voxel sizes (nm, z, y, x) of the session's raw levels, finest
    first (session/volume.py raw_voxel_sizes), kept per dataset: the models
    answer carries them, every two seconds while the tab waits for a model.
    Empty without a dataset, or when its levels cannot be read (logged)."""
    path = get_session().dataset_path
    if not path:
        return []
    try:
        return [list(size) for size in _cached_raw_voxel_sizes(path)]
    except Exception as e:
        logger.warning(f"Could not list the levels of {path}: {e}")
        return []


def shown_voxel_size():
    """The voxel size of the raw level the viewer shows now
    (portable/volume.py shown_level at the viewer's zoom, viewer/raw.py
    view_of), the finest without a viewer or a zoom: a paint-first volume's
    when New Volume is given none. A ValueError when the raw's levels
    cannot be read."""
    from cellmap_flow.viewer.raw import view_of

    sizes = session_raw_voxel_sizes()
    viewer = get_session().viewer
    view = view_of(viewer.state) if viewer is not None else None
    return list(sizes[portable_volume.shown_level(sizes, view.pixel_nm if view else None)])


def paint_first_refusal(corrections_dir):
    """Training's submit refused for a session whose volume
    (session_store's session_volume of ``corrections_dir``) was painted
    before any model: ``(response, 400)``, whose error is portable/volume.py
    training_refusal's sentence (the browser's submit says the same) and
    whose ``paint_first`` the tab can key on. None when the volume can be
    trained on, or there is none (submit says so itself)."""
    _, volume = session_store().session_volume(str(corrections_dir))
    refusal = volume and portable_volume.training_refusal(volume)
    if not refusal:
        return None
    return jsonify({"success": False, "error": refusal, "paint_first": True}), 400


def _get_selected_model_config(model_name):
    if not get_session().models_config:
        return None, (jsonify({"success": False, "error": "No models loaded"}), 400)

    model_config = find_model_config(model_name)
    if model_config is None:
        return None, (
            jsonify({"success": False, "error": f"Model {model_name} not found"}),
            404,
        )

    return model_config, None


def _geometry_from_server(name):
    """Geometry from the running inference server, which already has the model."""
    return model_geometry(fetch_model_info(running_job_host(name)))


def _geometry_from_saved_pipeline(name):
    """The geometry the pipeline builder saved with ``name``'s entry, and the
    channel names the entry gives (a fly model's ``channels``); None without one."""
    cfg = get_session().builder_model_configs.get(name)
    if not cfg:
        return None
    output_channels = cfg.get("output_channels", 1)
    return {
        "write_shape": cfg.get("write_shape", []),
        "output_voxel_size": cfg.get("output_voxel_size", []),
        "output_channels": output_channels,
        "channels": output_channel_names(cfg, output_channels),
    }


def _declared_geometry(model_config):
    """What ``model_config`` declares of its geometry without being built
    (``ModelConfig.declared_geometry``), or None."""
    declared = getattr(model_config, "declared_geometry", None)
    return model_geometry(declared()) if callable(declared) else None


def listed_geometry(name, model_config):
    """``name``'s geometry as the models list has it: its server's, as it
    serves the model; else what the pipeline builder saved; else what the
    model declares (``model_config`` its config, None for a model a job
    serves without one). None when none of them knows it. Never the model
    built here (see get_finetune_models)."""
    return (
        _geometry_from_server(name)
        or _geometry_from_saved_pipeline(name)
        or _declared_geometry(model_config)
    )


def channel_pick_refusal(model_name, model_config, output_type, select_channel):
    """Training's submit refused for a model that trains one channel at a time
    when no channel is picked: ``(response, 400)``, with the sentence the
    browser's submit says too.

    Such a model is one with more than one output channel, trained on a
    target that is not one per channel (common.trains_one_channel, of the
    ``output_type`` submit resolved): the models list says
    ``channel_choice`` for it, and the training form then asks for the
    channel: without one, the job would train every channel on the paint.
    None when a channel is picked, or the model trains every channel at
    once, or the list has no geometry for it (the form shows no pick then,
    and the job trains as the command line does).
    """
    if select_channel is not None:
        return None
    geometry = listed_geometry(model_name, model_config)
    if not geometry or not trains_one_channel(output_type, geometry["output_channels"]):
        return None
    return jsonify({"success": False, "error": (
        f"{model_name} has {geometry['output_channels']} output channels and trains one at a time: "
        "pick the channel your paint is for (Channel, in the training form)."
    )}), 400


def volume_grid(volume) -> dict:
    """A volume record's grid (portable/volume.py GRID_FIELDS, the record's
    own keys), or None when the record lacks one of them."""
    grid = {key: volume.get(key) for key in portable_volume.GRID_FIELDS}
    return grid if all(grid.values()) else None


def grid_change(model_name, model_config, corrections_dir):
    """Whether training ``model_name`` on the session volume of
    ``corrections_dir`` takes its paint carried onto the model's grid first:
    ``(carry, refusal)``, with nothing changed either way.

    ``carry`` is None when the model trains on the volume as it is: the
    volume is the model's own, or its grid is the model's too
    (portable/volume.py same_grid: a finetune of its model, any model of the
    same sizes), or there is no volume with a grid to compare (one a
    dashboard before this one registered, which trains as it did). Else it
    is what ``carry_onto_model`` takes after the model's name and config,
    ``(geometry, from_id, from_volume)``. The model's grid is the volume a
    Create Volume for it would plan now, on the session's dataset; its sizes
    are its server's or the geometry cache's (geometry_cache.known_geometry),
    never the model built here (see rank_routes._served_geometry).
    ``refusal`` is ``(response, 400)`` when neither knows them yet: its
    server has not answered."""
    from_id, volume = session_store().session_volume(str(corrections_dir))
    from_grid = volume and volume_grid(volume)
    if not from_grid or volume.get("model_name") == model_name:
        return None, None
    geometry = known_geometry(model_name, model_config)
    if geometry is None:
        return None, (jsonify({"success": False, "error": (
            f"Your paint is on {volume.get('model_name')}'s grid, and {model_name}'s is not known yet: its server "
            f"has not answered with its sizes. Wait until {model_name} is running (the Models tab), then press "
            "Train again, which carries the paint onto its grid."
        )}), 400)
    session = get_session()
    planned = plan_volume(session.dataset_path, geometry, resample=session.resample)
    to_grid = {"input_size": planned.input_size, "output_size": planned.chunk_size,
               "input_voxel_size": planned.input_voxel_size, "output_voxel_size": planned.output_voxel_size}
    if portable_volume.same_grid(from_grid, to_grid):
        return None, None
    return (geometry, from_id, volume), None


def carry_onto_model(model_name, model_config, geometry, from_id, from_volume):
    """The paint of the session's volume ``from_id`` (``from_volume`` its
    record) carried onto ``model_name``'s grid, ``geometry`` its sizes
    (resolve_model_geometry's): a volume made for the model (make_volume)
    beside it in its corrections dir, the old one's last strokes pulled from
    MinIO first, the paint carried onto the new one (session/volume.py
    carry_paint) and mirrored to MinIO, and its paint layer in the viewer in
    the old one's place. The old volume is retired (finetune_utils
    retire_annotation_volume: out of the session, kept on disk in its
    corrections dir's ``superseded/``), so the session has the one volume,
    the new one, which its manifest (make_volume wrote it) trains on.

    Which model to start from's Use this model (rank_routes) and training's
    submit for a model on another grid (grid_change) both carry through
    here; neither the chain nor the channel shown changes here (Use this
    model sets those itself, submit leaves them as the user set them).
    Returns ``{"volume_id", "volume" (its record), "layer_name", "painted",
    "source", "conflicts", "chunks", "superseded"}``: carry_paint's counts,
    and where the old volume went."""
    session = get_session()
    corrections_dir = from_volume["corrections_dir"]
    sync_annotation_volume_from_minio(from_id)
    to_id, zarr_path, minio_url, _ = make_volume(model_name, model_config, geometry, corrections_dir)
    to_volume = session.annotation_volumes[to_id]
    carried = carry_paint(from_volume, to_volume)
    ensure_minio_serving(zarr_path, to_id, output_base_dir=corrections_dir)
    layer = f"annotation_{to_id}"
    if session.viewer is not None:
        with session.viewer.txn() as s:
            old = f"annotation_{from_id}"
            if old in s.layers:
                del s.layers[old]
        add_annotation_layer(session.viewer, layer, f"{minio_url}/annotation")
    superseded = retire_annotation_volume(from_id)
    refresh_annotated_regions_layer()
    logger.info(f"Carried {from_id}'s paint to {to_id} for {model_name}: {carried['painted']} voxels from {carried['source']}"
                + (f"; {carried['conflicts']} left unpainted where object and background met in one voxel" if carried["conflicts"] else "")
                + (f"; {from_id} kept in {superseded}" if superseded else ""))
    return {"volume_id": to_id, "volume": to_volume, "layer_name": layer, **carried, "superseded": superseded}


def _waiting(name):
    """Why ``name`` has no geometry yet: its server is not up, or it is not
    running at all (stopped on the Models tab, its config kept)."""
    from cellmap_flow.dashboard.services.launch import running_names

    return "waiting for its server" if name in running_names() else "not running"


def _finetune_modes(model_config):
    """What the model can be finetuned with ("lora", "full"), for the tab's
    LoRA rank options; None when there is no config to ask (a model a job
    serves without one), which leaves every option offered."""
    if model_config is None or not hasattr(model_config, "finetune_modes"):
        return None
    try:
        return list(model_config.finetune_modes())
    except Exception as e:
        logger.warning(f"Could not tell how {getattr(model_config, 'name', 'a model')} can be finetuned: {e}")
        return None


@finetune_bp.route("/api/finetune/models", methods=["GET"])
def get_finetune_models():
    """The models to finetune, and the raw's levels: ``{"models",
    "selected_model", "raw_voxel_sizes"}``.

    Each model is ``{"name", "write_shape" (nm), "output_voxel_size",
    "output_channels", "channels", "channel_choice", "finetune_modes"}``:
    ``channels`` the output channels' names (None when nothing names them),
    ``channel_choice`` whether training takes one of them at a time
    (common.channel_choice). One whose geometry nothing knows yet is
    ``{"name", "status", "finetune_modes"}`` instead, its ``status``
    "waiting for its server" or, stopped on the Models tab, "not running".
    """
    try:
        models = []
        seen = set()

        # Every name we might report on: configured models first, then any job
        # running without a matching config (a yaml-launched model, say).
        session = get_session()
        configs_by_name = {}
        for mc in session.models_config or []:
            name = getattr(mc, "name", None)
            if name:
                configs_by_name[name] = mc
        names = list(configs_by_name)
        for job in session.jobs or []:
            job_name = getattr(job, "model_name", None)
            if job_name and job_name not in configs_by_name:
                names.append(job_name)

        for name in names:
            if name in seen:
                continue
            # The server's first, as it serves the model; else what the
            # pipeline builder saved, or what the model declares. Never the
            # model built here: the tab asks every two seconds while it waits
            # for a server, and building it loads its weights and runs a
            # forward on the dashboard's CPU (25 s for a 3 GB TorchScript). A
            # model none of them knows is listed with why it has no geometry
            # yet, its ``status``, until its server answers.
            config = configs_by_name.get(name)
            geometry = listed_geometry(name, config)
            if geometry is None:
                geometry = {"status": _waiting(name)}
            else:
                # By the same rule, so without building it either.
                geometry["channel_choice"] = channel_choice(config, geometry["output_channels"], geometry["channels"])
            seen.add(name)
            models.append({"name": name, **geometry, "finetune_modes": _finetune_modes(config)})

        selected = models[0]["name"] if len(models) == 1 else None
        # The raw's levels, which New Volume offers as the voxel size of a
        # volume painted before any model.
        return jsonify({"models": models, "selected_model": selected, "raw_voxel_sizes": session_raw_voxel_sizes()})
    except Exception as e:
        logger.error(f"Error getting finetune models: {e}")
        return jsonify({"error": str(e)}), 500


@finetune_bp.route("/api/finetune/create-volume", methods=["POST"])
def create_annotation_volume():
    body, refused = parse(CreateVolume, request.get_json() or {})
    if refused:
        return refused
    if body.model_name is None:
        return _create_paint_volume(body)
    try:
        model_name = body.model_name
        output_path = body.output_path

        model_config, error_response = _get_selected_model_config(model_name)
        if error_response is not None:
            return error_response

        # Ask the running inference server for the geometry. It already has
        # the model; building it here instead costs a full load on the
        # dashboard's CPU -- 43s in one measured session, for shapes the
        # server can report in milliseconds -- and is what made "create
        # annotation volume" feel slow. model_config.config stays as the
        # fallback for when no server is up.
        config = resolve_model_geometry(model_name, model_config)

        if not get_session().dataset_path:
            return jsonify({"success": False, "error": "No dataset path configured"}), 400
        _, corrections_dir = ensure_corrections_storage(output_path)
        volume_id, zarr_path, minio_url, geometry = make_volume(model_name, model_config, config, corrections_dir)
        return jsonify(_created(volume_id, zarr_path, minio_url, geometry, model_name))
    except Exception as e:
        logger.error(f"Error creating annotation volume: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def _create_paint_volume(body):
    """create-volume without a model: a paint-first volume at the body's
    voxel size (make_paint_volume). A size the raw cannot be planned at is
    a 400 with the plan's sentence."""
    try:
        if not get_session().dataset_path:
            return jsonify({"success": False, "error": "No dataset path configured"}), 400
        _, corrections_dir = ensure_corrections_storage(body.output_path)
        volume_id, zarr_path, minio_url, geometry = make_paint_volume(body.voxel_size, corrections_dir)
        logger.info(f"New annotation volume {volume_id} for no model yet, at {list(geometry.output_voxel_size)} nm")
        return jsonify(_created(volume_id, zarr_path, minio_url, geometry, None))
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error creating annotation volume: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def _created(volume_id, zarr_path, minio_url, geometry, model_name):
    """create-volume's answer for a volume made: where it is, as the tab
    reaches it, and its grid (crops.js logs it). A paint-first volume
    claims no voxel size of a model's: its own is answered as the claimed."""
    return {
        "success": True,
        "volume_id": volume_id,
        "zarr_path": zarr_path,
        "minio_url": minio_url,
        "neuroglancer_url": f"{minio_url}/annotation",
        "model_name": model_name,
        "metadata": {
            "dataset_shape_voxels": list(geometry.dataset_shape_voxels),
            "chunk_size": list(geometry.chunk_size),
            "output_voxel_size": list(geometry.output_voxel_size),
            "claimed_output_voxel_size": list(geometry.claimed_output_voxel_size or geometry.output_voxel_size),
            "dataset_offset_nm": list(geometry.dataset_offset_nm),
        },
    }


@finetune_bp.route("/api/finetune/user-prefs", methods=["GET"])
def get_user_prefs():
    return jsonify({"success": True, "prefs": load_user_prefs()})


@finetune_bp.route("/api/finetune/user-prefs", methods=["POST"])
def set_user_prefs():
    data = request.get_json() or {}
    try:
        prefs = load_user_prefs()
        prefs.update({key: value for key, value in data.items() if value is not None})
        save_user_prefs(prefs)
        return jsonify({"success": True, "prefs": prefs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
