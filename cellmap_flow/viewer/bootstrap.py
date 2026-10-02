"""A new viewer on a dataset: ``new_viewer()``.

The viewers the dashboard serves are made here: the CLIs' startup viewer
(dashboard.services.startup.generate_neuroglancer_url), /api/set-data, and the
bounding-box tool. ``cellmap_flow view`` (cli/viewer_cli.py) still builds its
own. It never starts the dashboard.
"""

import logging

import neuroglancer

from cellmap_flow.image_data_interface import legacy_meta
from cellmap_flow.io import metadata
from cellmap_flow.viewer.layers import raw_layer

logger = logging.getLogger(__name__)


def raw_dimensions(dataset_path):
    """The viewer's dimensions for ``dataset_path``: the axes and voxel size
    of the finest level of its pyramid (of the array itself, if it is none).

    Set before any layer is added, so that the raw data decides the viewer's
    coordinate space rather than whichever layer neuroglancer takes it from,
    such as an extra layer at another voxel size. None if it cannot be read.
    """
    try:
        group = dataset_path
        last = dataset_path.rstrip("/").rsplit("/", 1)[-1]
        if last.startswith("s") and last[1:].isdigit():  # one level, as get_raw_layer reads it
            group = dataset_path.rstrip("/").rsplit("/", 1)[0]
        try:
            levels = [meta for _, meta in metadata.list_levels(group)]
        except Exception:
            levels = [metadata.read_array_meta(dataset_path)]
        finest = min(levels, key=lambda meta: tuple(meta.spatial().voxel_size))
        # The names and sizes ImageDataInterface gives the raw layer's volume.
        voxel_size, _, _, _, names, _ = legacy_meta(finest)
        return neuroglancer.CoordinateSpace(names=names, units="nm", scales=voxel_size)
    except Exception as e:
        logger.warning(f"Could not read the viewer's dimensions from {dataset_path}: {e}")
        return None


def new_viewer(dataset_path, *, scales=None, raw=None, raw_name="data", layers=None):
    """A viewer showing the raw data at ``dataset_path``, then ``layers``.

    - ``scales``: the viewer's dimensions, z, y, x in nm. None takes them from
      the raw (raw_dimensions), or leaves them to neuroglancer if it cannot
      be read.
    - ``raw``: the raw data's layer, if the caller built it (raw_layer);
      shown as ``raw_name``.
    - ``layers``: ``{name: layer}``, added after the raw in their order.

    Its server binds every interface: the browser is rarely on the machine
    the viewer runs on.
    """
    if scales is None:
        dimensions = raw_dimensions(dataset_path)
    else:
        dimensions = neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="nm", scales=list(scales))
    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        if dimensions is not None:
            s.dimensions = dimensions
        s.layers[raw_name] = raw_layer(dataset_path) if raw is None else raw
        for name, layer in (layers or {}).items():
            s.layers[name] = layer
    return viewer
