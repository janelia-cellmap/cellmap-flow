"""What cellmap-flow reads from each layout it supports, through worldarray.

Every reader (open_raw and RawInput, the viewer, blockwise, the finetune crop
loader) opens a path with ``io.metadata.open_level`` and takes voxel sizes,
corners and levels from worldarray's record, so one row per layout: OME-Zarr
v2 and v3 at every placement of their transforms, funlib and N5 attributes,
precomputed, http, s3 and gs. These rows are cellmap-flow's record of what it
relies on worldarray for. Corners are voxel 0's lower corner in nm; an OME
translation (worldarray's ``translation``) is voxel 0's centre.
"""

import functools
import http.server
import json
import os
import threading
import urllib.parse
from types import SimpleNamespace

import numpy as np
import pytest
import worldarray as wa

from cellmap_flow.io import paths
from cellmap_flow.io.metadata import open_level
from cellmap_flow.io.multiscale import closest_raw_scale

ZYX = ("z", "y", "x")
DATA = np.zeros((4, 4, 4), np.uint8)
V3 = "v3.zarr"


def _v3_group(path):
    """A zarr v3 group with no attributes; its path."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump({"zarr_format": 3, "node_type": "group", "attributes": {}}, f)
    return str(path)


def _v3_group_with_s0(f):
    f.write_array("zarr3", DATA, name="g.zarr/s0")
    return _v3_group(f.tmp / "g.zarr")


def _n5_level_in_multiscales(f):
    """An N5 level with a resolution and no offset, whose group's multiscales list it
    with a voxel size of their own (8 nm, not used) and an offset (x, y, z)."""
    import zarr
    from zarr.n5 import N5FSStore

    path = f.write_array("n5", np.zeros((10, 20, 30), np.uint8), {"resolution": [3, 2, 1]}, "m.n5/raw/s0")
    transform = {"scale": [8, 8, 8], "translate": [30, 20, 10], "units": ["nm"] * 3}
    zarr.open_group(N5FSStore(str(f.tmp / "m.n5")), mode="a")["raw"].attrs["multiscales"] = [
        {"datasets": [{"path": "s0", "transform": transform}]}
    ]
    return path


def _unlisted_level(f):
    """raw/s1 next to an OME level raw/s0 that raw's multiscales list alone; s1
    has its own funlib attributes."""
    f.ome_pyramid(((8, 0),), name="u.zarr/raw")
    return f.write_array("zarr2", DATA, {"resolution": [16] * 3, "offset": [0] * 3}, "u.zarr/raw/s1")


class _Buckets(http.server.SimpleHTTPRequestHandler):
    """A directory served as a web server, an S3 endpoint and GCS, by the
    first part of the request path, which names the "bucket":

    - ``/web/<key>``: plain http;
    - ``/bkt/<key>``: a public S3 bucket (path-style), which refuses a signed
      request as S3 refuses a public read signed with a key it doesn't know;
    - ``/private/<key>``: an S3 bucket that needs a signed request;
    - ``/storage/v1/b/<bucket>/o/<key>``: GCS's JSON API, which refuses the
      bucket "denied" (as when broken credentials are sent); ``/denied/<key>``
      is that bucket's public URL.

    Every bucket holds the same files. S3 needs an ETag on every object.
    """

    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("ETag", '"0"')
        super().end_headers()

    def _refuse(self):
        path = urllib.parse.urlparse(self.path).path
        signed = "Authorization" in self.headers
        return (
            (path.startswith("/bkt/") and signed)
            or (path.startswith("/private/") and not signed)
            or path.startswith("/storage/v1/b/denied/")
        )

    def send_head(self):
        if self._refuse():
            self.send_error(403)
            return None
        return super().send_head()

    def translate_path(self, path):
        path = urllib.parse.urlparse(path).path
        if path.startswith("/storage/v1/b/"):
            bucket, _, key = path[len("/storage/v1/b/"):].partition("/o/")
            path = f"/{bucket}/{urllib.parse.unquote(key)}"
        return super().translate_path("/" + path.lstrip("/").partition("/")[2])


@pytest.fixture
def at_url(tmp_path, monkeypatch):
    """``at_url(scheme, path, bucket=...)``: the URL serving a path under tmp_path,
    over "http", "s3" or "gs" (see _Buckets), with AWS credentials in the
    environment that the public bucket refuses."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Buckets, directory=str(tmp_path)))
    # Polled often, so that shutting it down takes milliseconds, not half a second.
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    for name, value in {"AWS_ENDPOINT_URL": endpoint, "AWS_REGION": "us-east-1", "AWS_ACCESS_KEY_ID": "AKIDEXAMPLE",
                        "AWS_SECRET_ACCESS_KEY": "secret", "TENSORSTORE_GCS_HTTP_URL": endpoint}.items():
        monkeypatch.setenv(name, value)

    def url(scheme, path, bucket=None):
        root = {"http": f"{endpoint}/web", "s3": "s3://" + (bucket or "bkt"), "gs": "gs://" + (bucket or "bkt")}[scheme]
        return path.replace(str(tmp_path), root, 1)

    yield url
    server.shutdown()
    server.server_close()


# layout: (how it is written, (format, axes, voxel size, corner, shape, chunk shape))
# An array that names no axes is z, y, x to worldarray, with c before them.
LAYOUTS = {
    # Janelia's levels share a corner at -4 nm: translation = scale / 2 - 4.
    "ome-v2-level": (
        lambda f: f.ome_pyramid(((8, 0), (16, 4))) + "/s1",
        ("zarr2", ZYX, (16.0,) * 3, (-4.0,) * 3, (8, 8, 8), (4, 4, 4)),
    ),
    "ome-v2-no-translation": (
        lambda f: f.ome_pyramid(((8, None),)) + "/s0",
        ("zarr2", ZYX, (8.0,) * 3, (-4.0,) * 3, (16, 16, 16), (8, 8, 8)),
    ),
    "ome-v2-channels": (
        lambda f: f.ome_pyramid(((8, 4),), shape=(8, 8, 8), channels=2) + "/s0",
        ("zarr2", ("c",) + ZYX, (1.0, 8.0, 8.0, 8.0), (0.0,) * 4, (2, 8, 8, 8), (1, 4, 4, 4)),
    ),
    # Each v3 level has its own zarr.json; the transform is the group's.
    "ome-v3-level": (
        lambda f: f.ome_pyramid(((4, (1, 2, 3)), (8, (1, 2, 3))), zarr_format=3) + "/s1",
        ("zarr3", ZYX, (8.0,) * 3, (-3.0, -2.0, -1.0), (8, 8, 8), (4, 4, 4)),
    ),
    "ome-v3-channels": (
        lambda f: f.ome_pyramid(((8, 4),), shape=(8, 8, 8), channels=2, zarr_format=3) + "/s0",
        ("zarr3", ("c",) + ZYX, (1.0, 8.0, 8.0, 8.0), (0.0,) * 4, (2, 8, 8, 8), (1, 4, 4, 4)),
    ),
    # A group's own path reads its first level.
    "ome-v3-group": (
        lambda f: f.ome_pyramid(((4, 2), (8, 4)), zarr_format=3),
        ("zarr3", ZYX, (4.0,) * 3, (0.0,) * 3, (16, 16, 16), (8, 8, 8)),
    ),
    "ome-float-voxel-size": (
        lambda f: f.ome_pyramid((((5.24, 4, 4), (2.62, 2, 2)),), shape=(200, 4, 4)) + "/s0",
        ("zarr2", ZYX, (5.24, 4.0, 4.0), (0.0,) * 3, (200, 4, 4), (100, 2, 2)),
    ),
    "ome-micrometer": (
        lambda f: f.ome_pyramid((((0.008, 0.004, 0.004), (0.08, 0.04, 0.04)),), unit="micrometer") + "/s0",
        ("zarr2", ZYX, (8.0, 4.0, 4.0), (76.0, 38.0, 38.0), (16, 16, 16), (8, 8, 8)),
    ),
    # A group's own path reads its first level, as a v3 group's does.
    "ome-v2-group": (
        lambda f: f.ome_pyramid(((8, 0), (16, 4))),
        ("zarr2", ZYX, (8.0,) * 3, (-4.0,) * 3, (16, 16, 16), (8, 8, 8)),
    ),
    # The multiscales are on em, the group above the level.
    "ome-v2-nested-level": (
        lambda f: f.ome_pyramid(((8, 0), (16, 4)), name="n.zarr/em") + "/s1",
        ("zarr2", ZYX, (16.0,) * 3, (-4.0,) * 3, (8, 8, 8), (4, 4, 4)),
    ),
    # A level its group's multiscales don't list is read from its own attributes,
    # not as the group's first level (8 nm).
    "ome-v2-unlisted-level": (_unlisted_level, ("zarr2", ZYX, (16.0,) * 3, (0.0,) * 3, (4, 4, 4), (2, 2, 2))),
    # A zarr v2 group without multiscales reads its s0.
    "funlib-group": (
        lambda f: f.write_array("zarr2", DATA, {"resolution": [8] * 3, "offset": [0] * 3}, "g.zarr/raw/s0")[:-3],
        ("zarr2", ZYX, (8.0,) * 3, (0.0,) * 3, (4, 4, 4), (2, 2, 2)),
    ),
    "funlib-attrs": (
        lambda f: f.raw_zarr(DATA, voxel_size=(8, 4, 4), offset=(80, 40, 40)),
        ("zarr2", ZYX, (8.0, 4.0, 4.0), (80.0, 40.0, 40.0), (4, 4, 4), (4, 4, 4)),
    ),
    # A v2 offset off the voxel grid is kept as written: stored numbers are never
    # rounded onto a grid (worldarray's spec, Conventions 4).
    "funlib-offset-off-grid": (
        lambda f: f.raw_zarr(DATA, offset=(4, 4, 4)),
        ("zarr2", ZYX, (8.0,) * 3, (4.0,) * 3, (4, 4, 4), (4, 4, 4)),
    ),
    "funlib-no-attrs": (
        lambda f: f.write_array("zarr2", DATA),
        ("zarr2", ZYX, (1.0,) * 3, (0.0,) * 3, (4, 4, 4), (2, 2, 2)),
    ),
    "funlib-root-array": (
        lambda f: f.write_array("zarr2", DATA, {"resolution": [8, 4, 4], "offset": [80, 40, 40]}, "root.zarr"),
        ("zarr2", ZYX, (8.0, 4.0, 4.0), (80.0, 40.0, 40.0), (4, 4, 4), (2, 2, 2)),
    ),
    # Fortran memory order is not an axis order.
    "funlib-f-order": (
        lambda f: f.write_array("zarr2", np.zeros((4, 6, 8), np.uint8), {"resolution": [8, 4, 2]}, order="F"),
        ("zarr2", ZYX, (8.0, 4.0, 2.0), (0.0,) * 3, (4, 6, 8), (2, 3, 4)),
    ),
    # Attributes covering the last axes: the others are channels.
    "funlib-channels": (
        lambda f: f.write_array("zarr2", np.zeros((3, 4, 4, 4), np.uint8), {"resolution": [8, 4, 4], "offset": [0] * 3}),
        ("zarr2", ("c",) + ZYX, (1.0, 8.0, 4.0, 4.0), (0.0,) * 4, (3, 4, 4, 4), (1, 2, 2, 2)),
    ),
    # A COSEM transform's translate is voxel 0's centre (worldarray's spec, Conventions 1).
    "v3-transform-attrs": (
        lambda f: f.write_array("zarr3", DATA, {"transform": {"scale": [8] * 3, "translate": [100, 200, 300]}}, V3),
        ("zarr3", ZYX, (8.0,) * 3, (96.0, 196.0, 296.0), (4, 4, 4), (2, 2, 2)),
    ),
    # A v3 array's own offset is taken as written.
    "v3-funlib-attrs": (
        lambda f: f.write_array("zarr3", DATA, {"resolution": [8] * 3, "offset": [4] * 3}, V3),
        ("zarr3", ZYX, (8.0,) * 3, (4.0,) * 3, (4, 4, 4), (2, 2, 2)),
    ),
    "v3-no-attrs": (
        lambda f: f.write_array("zarr3", DATA, name=V3),
        ("zarr3", ZYX, (1.0,) * 3, (0.0,) * 3, (4, 4, 4), (2, 2, 2)),
    ),
    "v3-group-without-multiscales": (
        _v3_group_with_s0,
        ("zarr3", ZYX, (1.0,) * 3, (0.0,) * 3, (4, 4, 4), (2, 2, 2)),
    ),
    "v3-empty-group": (lambda f: _v3_group(f.tmp / "g.zarr"), ValueError),
    # N5's attributes are x, y, z.
    "n5-resolution": (
        lambda f: f.write_array("n5", np.zeros((10, 20, 30), np.uint8), {"resolution": [3, 2, 1], "offset": [0] * 3}),
        ("n5", ZYX, (1.0, 2.0, 3.0), (0.0,) * 3, (10, 20, 30), (5, 10, 15)),
    ),
    # COSEM's N5: a C-order transform, whose translate is voxel 0's centre.
    "n5-transform": (
        lambda f: f.write_array("n5", np.zeros((10, 20, 30), np.uint8), {"transform": {
            "ordering": "C", "scale": [8, 4, 2], "translate": [80, 40, 20], "units": ["nm"] * 3}}),
        ("n5", ZYX, (8.0, 4.0, 2.0), (76.0, 38.0, 19.0), (10, 20, 30), (5, 10, 15)),
    ),
    # One unit for every axis, not a string to reverse with them ("um" read as "mu").
    "n5-units-string": (
        lambda f: f.write_array("n5", np.zeros((10, 20, 30), np.uint8), {
            "resolution": [3, 2, 1], "offset": [0] * 3, "units": "um"}),
        ("n5", ZYX, (1000.0, 2000.0, 3000.0), (0.0,) * 3, (10, 20, 30), (5, 10, 15)),
    ),
    # A voxel size without an offset is kept, at offset 0; it was once dropped for 1 nm.
    "n5-resolution-without-offset": (
        lambda f: f.write_array("n5", np.zeros((10, 20, 30), np.uint8), {"resolution": [3, 2, 1]}),
        ("n5", ZYX, (1.0, 2.0, 3.0), (0.0,) * 3, (10, 20, 30), (5, 10, 15)),
    ),
    # ... but a level its group's multiscales list takes that entry's transform, which
    # outranks the array's own resolution (worldarray's spec, Conventions 4); its
    # translate (C order) is the centre.
    "n5-offset-from-multiscales": (
        _n5_level_in_multiscales,
        ("n5", ZYX, (8.0, 8.0, 8.0), (26.0, 16.0, 6.0), (10, 20, 30), (5, 10, 15)),
    ),
    # BigDataViewer's and Paintera's N5; the offset (x, y, z) is not rounded onto the grid.
    "n5-pixel-resolution": (
        lambda f: f.write_array("n5", np.zeros((10, 20, 30), np.uint8), {
            "pixelResolution": {"dimensions": [2, 4, 8], "unit": "nm"},
            "downsamplingFactors": [2, 2, 1], "offset": [30, 20, 10]}),
        ("n5", ZYX, (8.0, 8.0, 4.0), (10.0, 20.0, 30.0), (10, 20, 30), (5, 10, 15)),
    ),
    # ... which usually have no offset: every level's corner is then at the origin, as
    # Neuroglancer draws N5 (worldarray 0.1.1; 0.1.0 followed BigDataViewer, each level
    # half a base voxel lower).
    "n5-pixel-resolution-without-offset": (
        lambda f: f.write_array("n5", np.zeros((10, 20, 30), np.uint8), {
            "pixelResolution": {"dimensions": [2, 4, 8], "unit": "nm"}, "downsamplingFactors": [2, 2, 1]}),
        ("n5", ZYX, (8.0, 8.0, 4.0), (0.0,) * 3, (10, 20, 30), (5, 10, 15)),
    ),
    # x, y, z voxels (3, 2, 1) at (4, 8, 16) nm: the corner is (16, 16, 12) nm. A volume's
    # one channel is dropped (tensorstore keeps it; worldarray's open_array squeezes it).
    "precomputed": (
        lambda f: f.write_array("precomputed", np.zeros((2, 10, 20), np.uint8), {
            "resolution": [4, 8, 16], "chunk_size": [10, 5, 2], "voxel_offset": [3, 2, 1]}),
        ("precomputed", ZYX, (16.0, 8.0, 4.0), (16.0, 16.0, 12.0), (2, 10, 20), (2, 5, 10)),
    ),
    # The scale is named by the last /s<N> only: not by /scans, as on /groups/scicompsoft.
    "precomputed-scale-under-an-s-directory": (
        lambda f: f.write_array("precomputed", np.zeros((2, 10, 20), np.uint8), {
            "resolution": [4, 8, 16], "chunk_size": [10, 5, 2], "voxel_offset": [3, 2, 1]}, "scans/pc") + "/s0",
        ("precomputed", ZYX, (16.0, 8.0, 4.0), (16.0, 16.0, 12.0), (2, 10, 20), (2, 5, 10)),
    ),
    "precomputed-scale": (
        lambda f: f.write_array("precomputed", np.zeros((2, 10, 20), np.uint8), {
            "resolution": [4, 8, 16], "chunk_size": [10, 5, 2], "voxel_offset": [3, 2, 1]}) + "/s0",
        ("precomputed", ZYX, (16.0, 8.0, 4.0), (16.0, 16.0, 12.0), (2, 10, 20), (2, 5, 10)),
    ),
}


@pytest.mark.parametrize("scheme", ["disk", "http", "s3", "gs"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_what_each_layout_reads_as(layout, scheme, tmp_path, ome_pyramid, raw_zarr, write_array, at_url):
    """On disk, and at each kind of URL (neuroglancer's precomputed://<url> for
    precomputed): a URL reads as the same files on disk do. ``grid`` is the
    opened level's record in nm; a unit conversion may leave float noise."""
    write, expected = LAYOUTS[layout]
    path = write(SimpleNamespace(tmp=tmp_path, ome_pyramid=ome_pyramid, raw_zarr=raw_zarr, write_array=write_array))
    if scheme != "disk":
        path = at_url(scheme, path)
    if expected is ValueError:
        with pytest.raises(ValueError, match="no levels found"):
            open_level(path)
        return
    fmt, axes, voxel_size, corner, shape, chunk_shape = expected
    meta = open_level(path).grid
    assert (meta.format, meta.axes, meta.shape, meta.chunk_shape) == (fmt, axes, shape, chunk_shape)
    assert meta.voxel_size == pytest.approx(voxel_size) and meta.corner == pytest.approx(corner)
    assert tuple(meta.axes[i] for i in meta.spatial) == ZYX


def test_the_full_record_of_a_channel_first_array(ome_pyramid):
    """worldarray's record: every axis in the array's order, the store's units
    ("" on a channel axis), the translation voxel 0's centre and ``corner`` its
    lower corner; the spatial axes are the ones named z, y, x."""
    path = ome_pyramid(((8, 4),), shape=(8, 8, 8), channels=2) + "/s0"
    meta = wa.metadata(path)
    assert (meta.format, meta.shape, meta.dtype, meta.chunk_shape, meta.fill_value) == (
        "zarr2", (2, 8, 8, 8), np.dtype("u1"), (1, 4, 4, 4), 0)
    assert (meta.axes, meta.units, meta.spatial, meta.placed_by) == (
        ("c",) + ZYX, ("", "nanometer", "nanometer", "nanometer"), (1, 2, 3), "ome")
    assert (meta.voxel_size, meta.translation, meta.corner) == ((1.0, 8.0, 8.0, 8.0), (0.0, 4.0, 4.0, 4.0), (0.0,) * 4)
    assert meta.in_nm() == meta and open_level(path).meta == meta


# --- levels --------------------------------------------------------------------


def _janelia(ome_pyramid, zarr_format):
    """A Janelia pyramid, 8, 16 and 32 nm, each level's corner at -4 nm."""
    return ome_pyramid(((8, 0), (16, 4), (32, 12)), shape=(32, 32, 32), zarr_format=zarr_format)


@pytest.mark.parametrize("zarr_format", [pytest.param(2, id="zarr2"), pytest.param(3, id="zarr3")])
def test_a_janelia_pyramids_levels_share_their_corner(ome_pyramid, zarr_format):
    """The group's document places every level (``placements``: the translation
    is voxel 0's centre); read as corners, the levels agree."""
    pyramid = wa.open_pyramid(_janelia(ome_pyramid, zarr_format), units="nanometer")
    assert pyramid.placements() == [
        ("s0", (8.0,) * 3, (0.0,) * 3), ("s1", (16.0,) * 3, (4.0,) * 3), ("s2", (32.0,) * 3, (12.0,) * 3),
    ]
    assert [meta.in_nm().corner for meta in pyramid.metas()] == [(-4.0,) * 3] * 3


def test_a_precomputed_volumes_levels_are_its_scales(write_array):
    """Named by their keys and opened at ``volume/<key>``; cellmap-flow's
    ``…/s<N>`` opens scale N, and belongs to the whole volume."""
    volume = write_array("precomputed", np.zeros((8,) * 3, np.uint8), {"resolution": [8] * 3}, scales=2)
    location = volume[len("precomputed://"):]
    pyramid = wa.open_pyramid(volume)
    assert pyramid.names == ["8_8_8", "16_16_16"]
    assert [(meta.path, meta.voxel_size) for meta in pyramid.metas()] == [
        (location + "/8_8_8", (8.0,) * 3), (location + "/16_16_16", (16.0,) * 3),
    ]
    assert wa.open_pyramid(volume + "/s1").names == pyramid.names
    assert open_level(volume + "/s1").voxel_size == (16.0,) * 3


@pytest.fixture(params=["zarr2", "zarr3", "precomputed"])
def pyramid(ome_pyramid, write_array, request):
    """Levels of 8, 16 and 32 nm: a Janelia pyramid's s0, s1 and s2, or a
    precomputed volume's scales, which are its levels."""
    if request.param == "precomputed":
        return write_array("precomputed", np.zeros((32,) * 3, np.uint8), {"resolution": [8] * 3}, scales=3)
    return _janelia(ome_pyramid, int(request.param[-1]))


def test_the_closest_raw_scale_from_the_group_or_one_of_its_levels(pyramid, tmp_path):
    assert closest_raw_scale(pyramid, (16, 16, 16)) == closest_raw_scale(pyramid + "/s0", (20,) * 3) == (16.0,) * 3
    assert closest_raw_scale(str(tmp_path / "missing.zarr"), (8, 8, 8)) is None, "undetermined"


def test_choosing_a_precomputed_scale_opens_none_of_them(write_array, monkeypatch):
    """One read of the volume's info gives every scale's voxel size; opening each
    scale took about 50 ms on gs://, and a volume there can have a dozen."""
    import tensorstore

    volume = write_array("precomputed", np.zeros((32,) * 3, np.uint8), {"resolution": [8] * 3}, scales=3)
    monkeypatch.setattr(tensorstore, "open", lambda *args, **kwargs: pytest.fail("a scale was opened"))
    assert closest_raw_scale(volume, (20,) * 3) == (16.0,) * 3


@pytest.mark.parametrize("scheme", ["http", "s3", "gs"])
def test_a_pyramid_at_a_url_opens_as_it_does_on_disk(pyramid, at_url, scheme):
    """Its levels and their placement, the level chosen for a voxel size, the
    level open_raw opens there and what a RawInput reads from it."""
    from cellmap_flow.io.raw import RawInput, open_raw, spatial_axes, spatial_shape

    def opened(path):
        levels = wa.open_pyramid(path, units="nanometer")
        raw = RawInput(open_raw(path, (16, 16, 16)), input_norms=[])
        array = raw.array
        return (
            levels.names, levels.placements(), [(meta.shape, meta.chunk_shape) for meta in levels.metas()],
            closest_raw_scale(path + "/s0", (20,) * 3),
            (array.level_name, array.voxel_size, array.corner, array.box, spatial_shape(array),
             array.meta.chunk_shape, spatial_axes(array)),
            raw.to_ndarray_ts(array.box).tolist(),
        )

    assert opened(at_url(scheme, pyramid)) == opened(pyramid)


@pytest.mark.parametrize("zarr_format", [pytest.param(2, id="zarr2"), pytest.param(3, id="zarr3")])
def test_a_missing_level_has_no_closest_scale(ome_pyramid, zarr_format):
    """Not the scale of the group above it, whose zarr.json a v3 lookup found."""
    missing = ome_pyramid(((8, 0), (16, 4)), zarr_format=zarr_format) + "/missing"
    assert closest_raw_scale(missing, (8, 8, 8)) is None


# --- where the files are -------------------------------------------------------------


def _url_rows(schemes, formats):
    return [pytest.param(scheme, fmt, None, id=f"{scheme}-{fmt}") for scheme in schemes for fmt in formats]


ARRAY_URLS = _url_rows(["http", "s3", "gs"], ["zarr2", "zarr3", "n5", "precomputed"]) + [
    # Credentials are used when a bucket needs them...
    pytest.param("s3", "zarr2", "private", id="s3-private-bucket"),
    # ...and a gs:// bucket is read at its public URL when GCS refuses them.
    pytest.param("gs", "precomputed", "denied", id="gs-refused-credentials"),
]


@pytest.mark.parametrize("scheme, fmt, bucket", ARRAY_URLS)
def test_an_array_at_a_url_reads_as_it_does_on_disk(write_array, at_url, scheme, fmt, bucket):
    def read(path):
        """The whole array, every axis, as stored."""
        array = open_level(path, units=None)
        return array.read_index((0,) * array.meta.ndim, array.meta.shape)

    path = write_array(fmt, np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8), name=V3 if fmt == "zarr3" else None)
    assert np.array_equal(read(at_url(scheme, path, bucket)), read(path))


_COSEM = "janelia-cosem-datasets/jrc_hela-2/jrc_hela-2"
_ERA5 = "gcp-public-data-arco-era5/ar/1959-2022-6h-64x32_equiangular_conservative.zarr/2m_temperature"
_S3 = {"driver": "s3", "bucket": "janelia-cosem-datasets", "aws_credentials": {"type": "anonymous"}}


def _cosem(suffix):
    return {**_S3, "path": f"jrc_hela-2/jrc_hela-2.{suffix}/"}


# path, the voxel size to open it at, (its voxel size, corner), the level as
# tensorstore opens it, and a voxel in tissue (z, y, x)
PUBLIC = [
    pytest.param("gs://flyem-male-cns/em/em-clahe-jpeg", (32,) * 3, ((32.0,) * 3, (0.0,) * 3),
                 {"driver": "neuroglancer_precomputed", "kvstore": "gs://flyem-male-cns/em/em-clahe-jpeg/",
                  "scale_index": 2}, (8960, 9984, 20736), id="gs-precomputed"),
    pytest.param("precomputed://https://storage.googleapis.com/flyem-male-cns/em/em-clahe-jpeg", (32,) * 3,
                 ((32.0,) * 3, (0.0,) * 3), {"driver": "neuroglancer_precomputed", "scale_index": 2,
                 "kvstore": "https://storage.googleapis.com/flyem-male-cns/em/em-clahe-jpeg/"},
                 (8960, 9984, 20736), id="https-precomputed"),
    pytest.param(f"s3://{_COSEM}.zarr/recon-1/em/fibsem-uint8", (10.48, 8, 8), ((10.48, 8.0, 8.0), (-2.62, -2.0, -2.0)),
                 {"driver": "zarr", "kvstore": {**_cosem("zarr/recon-1/em/fibsem-uint8/s1")}}, (1592, 400, 3000),
                 id="s3-ome-zarr"),
    pytest.param(f"https://{_COSEM.replace('/', '.s3.amazonaws.com/', 1)}.zarr/recon-1/em/fibsem-uint8", (10.48, 8, 8),
                 ((10.48, 8.0, 8.0), (-2.62, -2.0, -2.0)),
                 {"driver": "zarr", "kvstore": {**_cosem("zarr/recon-1/em/fibsem-uint8/s1")}}, (1592, 400, 3000),
                 id="https-ome-zarr"),
    # The level's COSEM transform translates to voxel 0's centre, (2.62, 2, 2): the corner
    # is where the zarr's is, not rounded onto the grid.
    pytest.param(f"s3://{_COSEM}.n5/em/fibsem-uint16/s1", None, ((10.48, 8.0, 8.0), (-2.62, -2.0, -2.0)),
                 {"driver": "n5", "kvstore": {**_cosem("n5/em/fibsem-uint16/s1")}}, (1592, 400, 3000), id="s3-n5"),
    pytest.param(f"gs://{_ERA5}", None, ((1.0,) * 3, (0.0,) * 3),
                 {"driver": "zarr", "kvstore": f"gs://{_ERA5}/"}, (1000, 0, 0), id="gs-zarr"),
]


@pytest.mark.network
@pytest.mark.parametrize("path, voxel_size, placed, level, voxel", PUBLIC)
def test_public_data_reads_as_tensorstore_reads_it(path, voxel_size, placed, level, voxel, tmp_path, monkeypatch):
    """With broken AWS and Google credentials configured, which must not stop a
    public read."""
    import tensorstore

    from cellmap_flow.io.raw import RawInput, open_raw, spatial_shape

    direct = tensorstore.open(level).result()
    if level["driver"] != "zarr":  # x, y, z(, channel) in tensorstore; worldarray reads them reversed
        direct = direct[tensorstore.d[:].transpose[::-1]][tensorstore.d[:].translate_to[0]]
    (tmp_path / "adc.json").write_text(json.dumps(
        {"type": "authorized_user", "client_id": "x", "client_secret": "y", "refresh_token": "z"}))
    for name, value in {"AWS_ACCESS_KEY_ID": "AKIAFAKEFAKEFAKEFAKE", "AWS_SECRET_ACCESS_KEY": "fake",
                        "GOOGLE_APPLICATION_CREDENTIALS": str(tmp_path / "adc.json")}.items():
        monkeypatch.setenv(name, value)
    array = open_raw(path, voxel_size)
    assert (array.voxel_size, array.corner) == placed
    assert spatial_shape(array) == tuple(direct.shape[-3:])
    size = (1,) * (direct.rank - 3) + tuple(min(64, s) for s in direct.shape[-3:])
    begin = (0,) * (direct.rank - 3) + voxel
    end = tuple(b + n for b, n in zip(begin, size))
    want = direct[tuple(slice(b, e) for b, e in zip(begin, end))].read().result()
    # Every axis worldarray has: it drops the single channel of a precomputed
    # volume, which tensorstore keeps.
    dropped = direct.rank - array.meta.ndim
    assert np.array_equal(array.read_index(begin[dropped:], end[dropped:]), want[(0,) * dropped])
    # Read in the world too: worldarray's exact box of those voxels (z from
    # voxel 1592's corner, 16681.54 nm, on OpenOrganelle) reads them.
    world = wa.Box.from_index(voxel, tuple(v + n for v, n in zip(voxel, size[-3:])), array.voxel_size,
                              array.translation)
    assert np.array_equal(RawInput(array, input_norms=[]).to_ndarray_ts(world), want[(0,) * (direct.rank - 3)])


@pytest.mark.parametrize("path, normalized", [
    pytest.param("/d/my\\ data.zarr", "/d/my data.zarr", id="local"),
    pytest.param("precomputed:///d/my\\ vol", "precomputed:///d/my vol", id="local-precomputed"),
    pytest.param("https://h/my\\ data.zarr", "https://h/my\\ data.zarr", id="https"),
    pytest.param("gs://b/my\\ vol", "gs://b/my\\ vol", id="gs"),
    pytest.param("precomputed://gs://b/my\\ vol", "precomputed://gs://b/my\\ vol", id="precomputed-url"),
])
def test_a_shell_escaped_space_is_a_space_on_disk_only(path, normalized):
    assert paths.normalize_path(path) == normalized
