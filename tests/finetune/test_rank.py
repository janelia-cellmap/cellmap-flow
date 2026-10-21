"""Which model to start from, in server mode (finetune/rank.py, session/volume.py
carry_paint, dashboard/routes/finetune/rank_routes.py), as the browser's
web/test/unit/finetune-rank.test.ts tests it there: the score is balanced
between object and background paint, read at the model's own cut-off and
forgiving a binary model the strokes' edges; a model running is read
through its server and any other started for its score alone; the raw is
stretched when that scores clearly better; the paint is carried onto
another model's grid.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from cellmap_flow.dashboard.routes.finetune import rank_routes, view_labels
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune import rank
from cellmap_flow.finetune.session.volume import carry_paint
from cellmap_flow.pipeline_spec import PipelineSpec
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


def test_only_a_narrow_raw_stretches_the_normalizer():
    norm = [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}, {"name": "LambdaNormalizer", "expression": "x*2-1"}]
    assert rank.windowed(norm, 100, 140) == [dict(norm[0], min_value=100.0, max_value=140.0), norm[1]]
    assert rank.windowed(norm, 0, 200) is None
    assert rank.windowed(norm[1:], 100, 140) is None


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


@pytest.fixture
def painted(tmp_path, monkeypatch):
    """The session's volume, a 16^3 grid painted object (2) at 2..6 and
    background (1) at 10..14 on each axis, and a model "model" whose logits
    are ``painted.logits`` (or ``painted.stretched``, read with the raw
    stretched): +4 in that object, -4 elsewhere, until a test sets them."""
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
        out = fake.stretched if input_norm is not None and fake.stretched is not None else fake.logits
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
        session.models_config = [SimpleNamespace(name="model")]
        session.jobs = [SimpleNamespace(model_name="model", host="http://gpu:8000")]
    else:
        def start_hosts(command, job_name, **kwargs):
            job = SimpleNamespace(model_name=job_name, host="http://gpu:9000", kill=lambda: killed.append(job_name))
            started.append(job)
            return job

        monkeypatch.setattr("cellmap_flow.models.registry.build_model", lambda entry, name: SimpleNamespace(name=name))
        monkeypatch.setattr("cellmap_flow.serving.launch.server_command", lambda config, path, resample: "serve")
        monkeypatch.setattr("cellmap_flow.jobs.launch.start_hosts", start_hosts)

    body = dashboard.post("/api/finetune/rank/score", json={
        "name": "model", "entry": {"type": "cellmap", "folder_path": "/m", "name": "model"}}).get_json()

    assert body["success"], body
    assert (body["matches"], body["cutoff"], body["channel"], body["window"]) == (1, 0, "nuc", None)
    assert (body["object_voxels"], body["background_voxels"]) == (64, 64)
    assert {host for host, _ in painted.reads} == {"http://gpu:8000" if running else "http://gpu:9000"}
    assert killed == ([] if running else ["model"]) and ("note" in body) != running


def test_a_narrow_raw_is_stretched_when_that_scores_clearly_better(dashboard, painted, monkeypatch):
    session = get_session()
    session.models_config = [SimpleNamespace(name="model")]
    session.jobs = [SimpleNamespace(model_name="model", host="http://gpu:8000")]
    session.set_pipeline(PipelineSpec([{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}], ()))
    monkeypatch.setattr(rank_routes, "_raw_window", lambda volume, boxes: (100.0, 120.0))
    # As normalized it sees nothing; with the raw stretched, the objects.
    painted.stretched, painted.logits = painted.logits, np.full_like(painted.logits, -4.0)

    body = dashboard.post("/api/finetune/rank/score", json={"name": "model"}).get_json()

    assert body["success"], body
    assert (body["matches"], body["matches_without_window"], body["window"]) == (1, 0.5, [100.0, 120.0])
    assert painted.reads[-1][1][0] == {"name": "MinMaxNormalizer", "min_value": 100.0, "max_value": 120.0}


def test_the_running_models_start_ticked_and_the_others_say_how_to_build_them(dashboard, painted, monkeypatch):
    session = get_session()
    session.models_config = [SimpleNamespace(name="model")]
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
    monkeypatch.setattr(launch, "_launch", lambda name, target, *args, daemon=False: launched.append(name))
    # The chain set, without a raw to redraw the viewer from.
    monkeypatch.setattr("cellmap_flow.dashboard.routes.pipeline._set_chain_and_redraw", lambda spec, url: session.set_pipeline(spec))
    monkeypatch.setattr("cellmap_flow.models.registry.build_model", lambda entry, name: SimpleNamespace(name=name))

    body = dashboard.post("/api/finetune/rank/use", json={
        "name": "mito_16nm", "window": [100, 120],
        "entry": {"type": "cellmap", "folder_path": "/models/mito", "name": "mito_16nm"}}).get_json()

    assert body["success"], body
    assert (body["volume_id"], body["from_volume"], body["window"]) == ("vol-2", "vol-1", [100.0, 120.0])
    # 4^3 object and 4^3 background voxels of 16 nm, 2^3 each at 32 nm.
    assert (body["painted_voxels"], body["source_voxels"], body["conflicts"]) == (16, 128, 0)
    assert launched == ["mito_16nm"] and [c.name for c in session.models_config][-1] == "mito_16nm"
    assert session.pipeline_spec.input_norm[0] == {"name": "MinMaxNormalizer", "min_value": 100.0, "max_value": 120.0}
    assert [layer.name for layer in viewer.state.layers] == ["annotation_vol-2"]
