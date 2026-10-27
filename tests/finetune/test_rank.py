"""Which model to start from, in server mode (finetune/rank.py, session/volume.py
carry_paint, dashboard/routes/finetune/rank_routes.py), as the browser's
web/test/unit/finetune-rank.test.ts tests it there: the score is balanced
between object and background paint, read at the model's own cut-off and
forgiving a binary model the strokes' edges; a binary model answers its
best channel by name and index, an affinity model neither; a model running is read
through its server and any other started for its score alone; the raw is
stretched when that scores clearly better; each row gives its threshold
and input window, what the other normalizations matched and the sigmoid
its layer needs; Use this model and Fit to Paint set the normalization and
the sigmoid in one chain set, Fit to Paint for the model seeding the paint
without a new volume; the paint is carried onto another model's grid; the
channel picked is shown, on the model's layer and its finetunes'.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from cellmap_flow.dashboard.routes.finetune import rank_routes, view_labels
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune import rank
from cellmap_flow.finetune.session.volume import carry_paint
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.portable.rank import best_try, display_postprocess, norm_tries, score_row
from cellmap_flow.serving.probe import UNBOUNDED


def _stripes(edge, called, forgive_edges, high=3.0, low=-3.0):
    """The score of a 1x4x10 block painted object in x < ``edge`` and
    background past it, by logits ``high`` in x < ``called`` and ``low`` past it."""
    labels = np.where(np.arange(10) < edge, 2, 1)[None, None].repeat(4, axis=1)
    logits = np.where(np.arange(10) < called, high, low)[None, None, None].repeat(4, axis=2).astype("f4")
    tallies = rank.empty_tallies(1)
    rank.tally(tallies, (labels > 1)[None], (labels > 0)[None], logits, rank.paint_edges(labels) if forgive_edges else None)
    return rank.rank_score(tallies, ["nuc"], False)


def test_the_score_is_balanced_read_at_the_models_cut_off_and_forgives_the_strokes_edges():
    right, all_background = _stripes(1, 1, False), _stripes(1, 0, False)
    assert right["matches"] == 1
    # Nine voxels in ten are background, all called right: half the score.
    assert all_background["matches"] == all_background["matches_at_zero"] == 0.5
    assert all_background["loss"] > right["loss"]
    # Objects at -1, background at -3: none called object at 0, all at its cut-off.
    shifted = _stripes(5, 5, False, high=-1.0)
    assert (shifted["matches_at_zero"], shifted["matches"]) == (0.5, 1)
    assert -3 <= shifted["cutoff"] < -1 and _stripes(5, 5, False)["cutoff"] == 0
    # A boundary a voxel off counts against it only on the strokes' edge.
    assert _stripes(5, 6, False)["matches"] < 1
    assert _stripes(5, 6, True)["matches"] == 1 and _stripes(5, 6, True)["scored_voxels"] == 32


def test_a_binary_model_answers_its_best_channel_by_index_and_an_affinity_model_none():
    # Three channels, the paint fitted by the second alone: its index is the
    # training form's Channel (select_channel), which Use this model sets.
    labels = np.where(np.arange(10) < 5, 2, 1)[None, None].repeat(4, axis=1)
    fits = np.where(np.arange(10) < 5, 3.0, -3.0)[None, None].repeat(4, axis=1)
    logits = np.stack([-fits, fits, np.full_like(fits, -3.0)]).astype("f4")
    tallies = rank.empty_tallies(3)
    rank.tally(tallies, np.repeat((labels > 1)[None], 3, axis=0), np.repeat((labels > 0)[None], 3, axis=0), logits)
    binary = rank.rank_score(tallies, ["all_mem", "mito", "er"], False)
    assert (binary["channel"], binary["channel_index"], binary["matches"]) == ("mito", 1, 1)
    # Every channel's match, the best first, to say why that one.
    assert [(c["channel"], c["channel_index"]) for c in binary["channel_matches"]][0] == ("mito", 1)
    assert len(binary["channel_matches"]) == 3
    affinity = rank.rank_score(tallies, ["z", "y", "x"], True)
    assert (affinity["channel"], affinity["channel_index"]) == (None, None)


def test_only_a_narrow_raw_stretches_the_normalizer():
    norm = [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}, {"name": "LambdaNormalizer", "expression": "x*2-1"}]
    assert rank.windowed(norm, 100, 140) == [dict(norm[0], min_value=100.0, max_value=140.0), norm[1]]
    assert rank.windowed(norm, 0, 200) is None
    assert rank.windowed(norm[1:], 100, 140) is None


def test_the_row_keeps_a_clearly_better_stretch_and_gives_the_threshold_and_input_window():
    norm = [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}]
    base = {"matches": 0.6, "cutoff": 0.0}
    # Two points better is not enough: the model's own range, at its own boundary.
    close = score_row(base, {"matches": 0.62, "cutoff": 1.0}, norm, (100, 120))
    assert (close["matches"], close["window"], close["threshold"], close["input_window"]) == (0.6, None, 0.5, [0.0, 255.0])
    # Four are: the stretched raw's range, at the stretched score's cut-off as a probability.
    clear = score_row(base, {"matches": 0.64, "cutoff": 1.0}, norm, (100, 120))
    assert (clear["matches"], clear["matches_without_window"], clear["window"]) == (0.64, 0.6, [100.0, 120.0])
    assert (clear["threshold"], clear["input_window"]) == (round(1 / (1 + math.exp(-1)), 3), [100.0, 120.0])
    assert score_row(base, None, [], None)["input_window"] is None
    assert score_row(None, None, norm, None) is None
    # The steps it was read through, stretched when the stretch is kept.
    assert (close["input_norm"], clear["input_norm"]) == (norm, [dict(norm[0], min_value=100.0, max_value=120.0)])


def test_a_model_is_tried_through_its_own_normalization_and_the_sessions():
    own = [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}, {"name": "LambdaNormalizer", "expression": "x*2-1"}]
    # The session's chain as the page keeps it, in strings: the same steps, tried once.
    same = [{"name": "MinMaxNormalizer", "min_value": "0", "max_value": "255"}, {"name": "LambdaNormalizer", "expression": "x*2-1"}]
    assert [t["from"] for t in norm_tries(own, "convention", same)] == ["convention"]
    assert [t["from"] for t in norm_tries(own, "declared", own[:1])] == ["declared", "session"]
    # What the others matched goes into the row, to say why this one.
    row = score_row({"matches": 0.9, "cutoff": 0.0}, None, own, None, "assumed", [{"from": "none", "matches": 0.51}])
    assert row["input_norm_others"] == [{"from": "none", "matches": 0.51}]
    # An assumption comes after the session's, so it loses a tie to it.
    assert [t["from"] for t in norm_tries(own, "assumed", own[:1])] == ["session", "assumed"]
    assert norm_tries([], "convention", []) == [{"input_norm": [], "from": "none"}]
    # The better is kept, the model's own among equals.
    assert best_try([{"matches": 0.5}, {"matches": 0.9}]) == 1
    assert best_try([{"matches": 0.7}, {"matches": 0.7}]) == 0
    assert best_try([None, None]) is None


def _volume(path, voxel, shape, chunks):
    zarr.open_group(str(path), mode="w").require_group("annotation").create_dataset(
        "s0", shape=shape, chunks=chunks, dtype="u1", fill_value=0)
    return {"zarr_path": str(path), "output_voxel_size": [voxel] * 3, "dataset_offset_nm": [voxel / 2] * 3}


@pytest.mark.parametrize("finer", [True, False], ids=["onto-a-finer-grid", "onto-a-coarser-grid"])
def test_paint_is_carried_onto_another_models_grid(tmp_path, finer):
    if finer:
        old = _volume(tmp_path / "old.zarr", 32, (4,) * 3, (4,) * 3)
        new = _volume(tmp_path / "new.zarr", 16, (8,) * 3, (4,) * 3)
        zarr.open(old["zarr_path"])["annotation/s0"][1, 1, 1] = 2
        expected = {"painted": 8, "source": 1, "conflicts": 0, "chunks": 1}
    else:
        # Chunks of 3 old voxels: new voxel 1 takes old voxels 2 and 3, in two chunks.
        old = _volume(tmp_path / "old.zarr", 16, (8,) * 3, (3,) * 3)
        new = _volume(tmp_path / "new.zarr", 32, (4,) * 3, (2,) * 3)
        labels = zarr.open(old["zarr_path"])["annotation/s0"]
        labels[2, 2, 2] = labels[3, 3, 3] = 2
        labels[6, 6, 6], labels[7, 7, 7] = 2, 1  # object against background in new voxel 3
        expected = {"painted": 1, "source": 4, "conflicts": 1, "chunks": 1}

    assert carry_paint(old, new) == expected
    carried = zarr.open(new["zarr_path"])["annotation/s0"][:]
    if finer:
        assert (carried[2:4, 2:4, 2:4] == 2).all() and carried.sum() == 16
    else:
        assert carried[1, 1, 1] == 2 and carried.sum() == 2


def _model(name):
    """A model config that declares no input normalization (ModelConfig.declared_input_norm), which ranking asks."""
    return SimpleNamespace(name=name, declared_input_norm=lambda data_range=None: None)


@pytest.fixture
def painted(tmp_path, monkeypatch):
    """The session's volume, a 16^3 grid painted object (2) at 2..6 and
    background (1) at 10..14 on each axis, and a model "model" whose logits
    are ``painted.logits`` (or ``painted.stretched``, read with the raw
    stretched to 100..120): +4 in that object, -4 elsewhere, until a test
    sets them."""
    path = tmp_path / "vol-1.zarr"
    labels = zarr.open_group(str(path), mode="w").require_group("annotation").create_dataset(
        "s0", shape=(16,) * 3, chunks=(8,) * 3, dtype="u1", fill_value=0)
    labels[2:6, 2:6, 2:6] = 2
    labels[10:14, 10:14, 10:14] = 1
    session = get_session()
    session.dataset_path = str(tmp_path / "raw.zarr")
    session.annotation_volumes["vol-1"] = {
        "zarr_path": str(path), "corrections_dir": str(tmp_path), "model_name": "model",
        "output_voxel_size": [16] * 3, "dataset_offset_nm": [8.0] * 3,
    }
    logits = np.full((1, 16, 16, 16), -4.0, "f4")
    logits[:, 2:6, 2:6, 2:6] = 4.0
    fake = SimpleNamespace(logits=logits, stretched=None, reads=[])

    def read_prediction(host, name, volume, lo, hi, info, postprocess=(), input_norm=None):
        fake.reads.append((host, input_norm))
        stretched = any(step.get("min_value") == 100.0 for step in input_norm or [])
        out = fake.stretched if stretched and fake.stretched is not None else fake.logits
        return out[(slice(None),) + tuple(slice(a, b) for a, b in zip(lo, hi))], lo, hi

    monkeypatch.setattr(view_labels, "read_prediction", read_prediction)
    monkeypatch.setattr(rank_routes, "sync_annotation_volume_from_minio", lambda volume_id: None)
    monkeypatch.setattr(rank_routes, "fetch_model_info", lambda host: {"channels": ["nuc"], "output_class": UNBOUNDED})
    monkeypatch.setattr(rank_routes, "autodetect_output_type", lambda config, output_type, offsets: ("binary", None))
    monkeypatch.setattr(rank_routes, "_raw_window", lambda volume, boxes: None)
    return fake


@pytest.mark.parametrize("running", [True, False], ids=["read-through-its-server", "started-for-its-score"])
def test_a_model_is_scored_on_the_paint(dashboard, painted, monkeypatch, running):
    session, started, killed = get_session(), [], []
    if running:
        session.models_config = [_model("model")]
        session.jobs = [SimpleNamespace(model_name="model", host="http://gpu:8000")]
    else:
        def start_hosts(command, job_name, **kwargs):
            job = SimpleNamespace(model_name=job_name, host="http://gpu:9000", kill=lambda: killed.append(job_name))
            started.append(job)
            return job

        monkeypatch.setattr("cellmap_flow.models.registry.build_model", lambda entry, name: _model(name))
        monkeypatch.setattr("cellmap_flow.serving.launch.server_command", lambda config, path, resample: "serve")
        monkeypatch.setattr("cellmap_flow.jobs.launch.start_hosts", start_hosts)

    body = dashboard.post("/api/finetune/rank/score", json={
        "name": "model", "entry": {"type": "cellmap", "folder_path": "/m", "name": "model"}}).get_json()

    assert body["success"], body
    assert (body["matches"], body["cutoff"], body["channel"], body["window"]) == (1, 0, "nuc", None)
    assert (body["object_voxels"], body["background_voxels"]) == (64, 64)
    # Its logits show right through a sigmoid, which the session's empty chain lacks.
    assert body["postprocess"] == [{"name": "SigmoidPostprocessor"}]
    assert {host for host, _ in painted.reads} == {"http://gpu:8000" if running else "http://gpu:9000"}
    assert killed == ([] if running else ["model"]) and ("note" in body) != running


def test_a_model_is_read_through_its_own_input_normalization_when_the_session_has_none(dashboard, painted, monkeypatch):
    """fly_organelles, ranked with no input normalization set, read raw 0 to
    255 and matched 51%, nucleus its best channel for paint of mitochondria.
    A model is read through what its start sets, and the session's too when
    that is another, the better kept."""
    session = get_session()
    session.models_config = [_model("model")]
    session.jobs = [SimpleNamespace(model_name="model", host="http://gpu:8000")]
    own = [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}, {"name": "LambdaNormalizer", "expression": "x*2-1"}]
    monkeypatch.setattr("cellmap_flow.dashboard.routes.pipeline.start_input_norm", lambda config, dtype, range_: (own, False))
    monkeypatch.setattr("cellmap_flow.dashboard.routes.model_advice.raw_dtype_and_range", lambda path: ("uint8", (0.0, 255.0)))
    # It sees the objects only through its own steps.
    original = view_labels.read_prediction

    def read(host, name, volume, lo, hi, info, postprocess=(), input_norm=None):
        out, plo, phi = original(host, name, volume, lo, hi, info, postprocess, input_norm)
        return (out if input_norm == own else np.full_like(out, -4.0)), plo, phi

    monkeypatch.setattr(view_labels, "read_prediction", read)

    body = dashboard.post("/api/finetune/rank/score", json={"name": "model"}).get_json()
    assert body["success"], body
    assert (body["matches"], body["input_norm_from"], body["input_norm"]) == (1, "convention", own)
    assert all(steps == own for _, steps in painted.reads)

    # With the session's chain as something else that sees nothing, its own
    # is still kept, and the row says what the other matched.
    session.set_pipeline(PipelineSpec([{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}], ()))
    body = dashboard.post("/api/finetune/rank/score", json={"name": "model"}).get_json()
    assert (body["matches"], body["input_norm_from"]) == (1, "convention")
    assert body["input_norm_others"] == [{"from": "session", "matches": 0.5}]


def test_a_model_nothing_declares_is_read_through_the_assumed_normalization(dashboard, painted, monkeypatch):
    """fly_organelles' /nrs copies declare nothing and are not CellMap's by
    their folder, so a start sets nothing; they were ranked on raw values
    ("Input none"). The model advice's assumption is tried instead."""
    session = get_session()
    session.models_config = [_model("model")]
    session.jobs = [SimpleNamespace(model_name="model", host="http://gpu:8000")]
    monkeypatch.setattr("cellmap_flow.dashboard.routes.model_advice.raw_dtype_and_range", lambda path: ("uint8", (0.0, 255.0)))

    body = dashboard.post("/api/finetune/rank/score", json={"name": "model"}).get_json()

    assert body["success"], body
    assert body["input_norm_from"] == "assumed"
    assert [s["name"] for s in body["input_norm"]] == ["MinMaxNormalizer", "LambdaNormalizer"]
    assert body["input_norm"][1]["expression"] == "x*2-1"


def test_a_narrow_raw_is_stretched_when_that_scores_clearly_better(dashboard, painted, monkeypatch):
    session = get_session()
    session.models_config = [_model("model")]
    session.jobs = [SimpleNamespace(model_name="model", host="http://gpu:8000")]
    session.set_pipeline(PipelineSpec([{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}], ()))
    monkeypatch.setattr(rank_routes, "_raw_window", lambda volume, boxes: (100.0, 120.0))
    # As normalized it sees nothing; with the raw stretched, the objects.
    painted.stretched, painted.logits = painted.logits, np.full_like(painted.logits, -4.0)

    body = dashboard.post("/api/finetune/rank/score", json={"name": "model"}).get_json()

    assert body["success"], body
    assert (body["matches"], body["matches_without_window"], body["window"]) == (1, 0.5, [100.0, 120.0])
    assert (body["threshold"], body["input_window"]) == (0.5, [100.0, 120.0])
    assert painted.reads[-1][1][0] == {"name": "MinMaxNormalizer", "min_value": 100.0, "max_value": 120.0}


def test_fit_to_paint_sets_the_input_normalization_and_sigmoid_of_the_model_seeding_without_a_new_volume(dashboard, painted, monkeypatch):
    session = get_session()
    session.models_config = [_model("model")]
    session.jobs = [SimpleNamespace(model_name="model", host="http://gpu:8000")]
    session.set_pipeline(PipelineSpec([{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}], ()))
    monkeypatch.setattr(rank_routes, "_raw_window", lambda volume, boxes: (100.0, 120.0))
    # The chain set, without a raw to redraw the viewer from.
    chains = []
    monkeypatch.setattr("cellmap_flow.dashboard.routes.pipeline._set_chain_and_redraw",
                        lambda spec, url: (chains.append(spec), session.set_pipeline(spec)))
    # As normalized it sees nothing; with the raw stretched, the objects at -1
    # over the rest at -4, so a cut-off under the model's own fits them.
    painted.stretched = np.where(painted.logits > 0, -1.0, -4.0).astype("f4")
    painted.logits = np.full_like(painted.logits, -4.0)
    volumes = dict(session.annotation_volumes)

    scored = dashboard.post("/api/finetune/rank/score", json={"name": "model"}).get_json()
    fitted = dashboard.post("/api/finetune/rank/input-norm", json={
        "input_norm": scored["input_norm"], "postprocess": scored["postprocess"]}).get_json()

    assert scored["success"] and fitted["success"], (scored, fitted)
    assert scored["cutoff"] < 0 and scored["threshold"] == round(1 / (1 + math.exp(-scored["cutoff"])), 3)
    assert scored["window"] == scored["input_window"] == [100.0, 120.0]
    stretched = {"name": "MinMaxNormalizer", "min_value": 100.0, "max_value": 120.0}
    assert fitted["input_norm"] == scored["input_norm"] == [stretched]
    # The sigmoid its logits need, set with the normalization in one chain set.
    sigmoid = {"name": "SigmoidPostprocessor"}
    assert fitted["postprocess"] == scored["postprocess"] == [sigmoid] and len(chains) == 1
    assert session.pipeline_spec.to_json_data() == {"input_norm": [stretched], "postprocess": [sigmoid]}
    assert session.annotation_volumes == volumes and rank_routes.session_store().session_volume()[0] == "vol-1"
    # Steps the dashboard has no op for change nothing.
    for body in ({"input_norm": [{"name": "NoSuchNormalizer"}]},
                 {"input_norm": [], "postprocess": [{"name": "NoSuchPostprocessor"}]}):
        assert not dashboard.post("/api/finetune/rank/input-norm", json=body).get_json()["success"]
    assert session.pipeline_spec.to_json_data() == {"input_norm": [stretched], "postprocess": [sigmoid]}


def test_the_running_models_start_ticked_and_the_others_say_how_to_build_them(dashboard, painted, monkeypatch):
    session = get_session()
    session.models_config = [_model("model")]
    session.jobs = [SimpleNamespace(model_name="model", host="http://gpu:8000")]
    session.model_catalog = {"mito": {"mito_16nm": "/models/mito"}}
    monkeypatch.setattr("cellmap_flow.models.hf_catalog.list_huggingface_models",
                        lambda: {"org/nuc-model": {"channels_names": ["nuc"], "output_voxel_size": [8, 8, 8]}})

    body = dashboard.get("/api/finetune/rank/candidates").get_json()

    assert body["volume"] == {"id": "vol-1", "model_name": "model", "painted_chunks": 2}
    rows = {c["name"]: c for c in body["candidates"]}
    assert [(n, c["ticked"], c["volume_model"]) for n, c in rows.items()] == [
        ("model", True, True), ("mito_16nm", False, False), ("nuc_model", False, False)]
    assert rows["mito_16nm"]["entry"] == {"type": "cellmap", "folder_path": "/models/mito", "name": "mito_16nm"}
    assert rows["nuc_model"]["entry"] == {"type": "huggingface", "repo": "org/nuc-model", "name": "nuc_model"}


def test_use_this_model_starts_it_and_carries_the_paint_to_a_volume_of_its_own(dashboard, painted, monkeypatch, viewer, tmp_path):
    from cellmap_flow.dashboard import finetune_utils
    from cellmap_flow.dashboard.routes.finetune import annotation_core
    from cellmap_flow.dashboard.routes.finetune.overlay import add_annotation_layer
    from cellmap_flow.dashboard.services import launch
    from cellmap_flow.models import geometry_cache

    session, launched = get_session(), []
    session.set_pipeline(PipelineSpec([{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}], ()))
    add_annotation_layer(viewer, "annotation_vol-1", "http://minio/vol-1.zarr/annotation")

    def make_volume(name, config, geometry, corrections_dir):
        session.annotation_volumes["vol-2"] = dict(_volume(tmp_path / "vol-2.zarr", 32, (8,) * 3, (4,) * 3),
                                                   model_name=name, corrections_dir=corrections_dir)
        return "vol-2", str(tmp_path / "vol-2.zarr"), "http://minio/vol-2.zarr", None

    monkeypatch.setattr(annotation_core, "make_volume", make_volume)
    monkeypatch.setattr(finetune_utils, "ensure_minio_serving", lambda *a, **k: None)
    monkeypatch.setattr(geometry_cache, "resolve_model_geometry", lambda name, config: {})
    # The chain set when the model is started, without a raw to redraw the viewer from.
    monkeypatch.setattr(launch, "_launch", lambda name, target, *args, daemon=False: launched.append(
        (name, session.pipeline_spec.to_json_data())))
    monkeypatch.setattr("cellmap_flow.dashboard.routes.pipeline._set_chain_and_redraw", lambda spec, url: session.set_pipeline(spec))
    monkeypatch.setattr("cellmap_flow.models.registry.build_model", lambda entry, name: _model(name))

    stretched, sigmoid = {"name": "MinMaxNormalizer", "min_value": 100.0, "max_value": 120.0}, {"name": "SigmoidPostprocessor"}
    body = dashboard.post("/api/finetune/rank/use", json={
        "name": "mito_16nm", "input_norm": [stretched], "postprocess": [sigmoid],
        "entry": {"type": "cellmap", "folder_path": "/models/mito", "name": "mito_16nm"}}).get_json()

    assert body["success"], body
    assert (body["volume_id"], body["from_volume"]) == ("vol-2", "vol-1")
    assert (body["input_norm"], body["postprocess"]) == ([stretched], [sigmoid])
    # 4^3 object and 4^3 background voxels of 16 nm, 2^3 each at 32 nm.
    assert (body["painted_voxels"], body["source_voxels"], body["conflicts"]) == (16, 128, 0)
    # Served through both.
    assert launched == [("mito_16nm", {"input_norm": [stretched], "postprocess": [sigmoid]})]
    assert [c.name for c in session.models_config][-1] == "mito_16nm"
    assert [layer.name for layer in viewer.state.layers] == ["annotation_vol-2"]


def test_the_sigmoid_a_ranked_models_layer_needs_and_nothing_else():
    """Use this model sets the sigmoid a model's output needs to show right:
    one for logits when nothing is set, none on top of an output in [0, 1]
    or [-1, 1]. Anything else, an affinity model's watershed among it, stays
    the Processing tab's suggestion."""
    sigmoid, threshold = {"name": "SigmoidPostprocessor"}, {"name": "ThresholdPostprocessor", "threshold": 0.5}
    assert display_postprocess("unbounded", [], False) == [sigmoid]
    assert display_postprocess("unbounded", [threshold], False) is None
    assert display_postprocess("unit", [sigmoid, threshold], False) == [threshold]
    assert display_postprocess("signed_unit", [sigmoid], False) == []
    assert display_postprocess("unit", [], False) is None
    assert display_postprocess("unbounded", [], True) is None


def test_the_channel_picked_is_shown_now_and_by_the_layers_of_the_model_and_its_finetunes_built_after(
        dashboard, viewer, monkeypatch, make_job):
    """fly_organelles' layer opened on all_mem, the first of its eight
    channels, while the paint and the training were for mito."""
    from cellmap_flow.dashboard import finetune_layers
    from cellmap_flow.dashboard.routes.finetune import channel as channel_routes
    from cellmap_flow.viewer.layers import prediction_layer

    info = {"output_channels": 8, "has_channel": True,
            "channels": ["all_mem", "organelle", "mito", "er", "nucleus", "pm", "vs", "ld"]}
    monkeypatch.setattr(channel_routes, "fetch_model_info", lambda host: info)
    monkeypatch.setattr(finetune_layers, "fetch_model_info", lambda host: info)
    session = get_session()
    session.jobs = [SimpleNamespace(model_name="m", host="http://gpu:8000")]
    with viewer.txn() as s:
        s.layers["m"] = prediction_layer("m", "http://gpu:8000", "blob", dataset_path=None, postprocess=[], info=info)

    def position(name):
        return viewer.state.layers[name].to_json().get("localPosition")

    body = dashboard.post("/api/finetune/channel", json={"model_name": "m", "select_channel": 2}).get_json()
    assert body == {"success": True, "shown": True} and position("m") == [2.5]
    # A finetune of it opens on the same channel.
    finetune_layers.add_finetuned_layer(make_job(inference_server_url="http://gpu:9000"), "m_finetuned_1")
    assert position("m_finetuned_1") == [2.5]
    # Another pick for it moves its running finetune's layer at once too, as the browser's does.
    session.jobs = [*session.jobs, SimpleNamespace(model_name="m_finetuned_1", host="http://gpu:9000")]
    dashboard.post("/api/finetune/channel", json={"model_name": "m", "select_channel": 3})
    assert position("m_finetuned_1") == [3.5]
    # A pick forgotten is not shown again; a channel that is no index is refused.
    assert dashboard.post("/api/finetune/channel", json={"model_name": "m", "select_channel": None}).get_json() == {
        "success": True, "shown": False}
    assert session.shown_channel("m_finetuned_1") is None
    assert dashboard.post("/api/finetune/channel", json={"model_name": "m", "select_channel": "mito"}).status_code == 400
