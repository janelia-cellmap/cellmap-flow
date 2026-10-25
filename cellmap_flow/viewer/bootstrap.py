"""A new viewer on a dataset: ``new_viewer()``.

The viewers the dashboard serves are made here: the CLIs' startup viewer
(dashboard.services.startup.generate_neuroglancer_url), ``cellmap_flow view``'s
(cli/viewer_cli.py), /api/set-data, and the bounding-box tool. It never
starts the dashboard. ``viewer_url()`` is the address a browser is given for
any of them.
"""

import ipaddress
import logging
from urllib.parse import urlparse

import neuroglancer
import worldarray as wa

from cellmap_flow.utils.address import this_machine_ip
from cellmap_flow.viewer import VIEWER_AXES
from cellmap_flow.viewer.layers import raw_layer

logger = logging.getLogger(__name__)

# Neuroglancer's GPU memory limit for the chunks it keeps, twice its default
# of 1 GB: a few Cellpose chunks (8 x 512 x 512, three float32 channels) fill
# 1 GB, and panning back then fetched them again. Changeable in the viewer's
# settings panel.
GPU_MEMORY_LIMIT = 2_000_000_000


def viewer_space(names, scales):
    """A viewer's coordinate space over axes ``names`` at ``scales`` nm, put
    in VIEWER_AXES order (x, y, z, then any others) when it has all three.
    """
    names, scales = list(names), list(scales)
    if all(axis in names for axis in VIEWER_AXES):
        order = [names.index(axis) for axis in VIEWER_AXES]
        order += [i for i in range(len(names)) if i not in order]
        names, scales = [names[i] for i in order], [scales[i] for i in order]
    return neuroglancer.CoordinateSpace(names=names, units="nm", scales=scales)


def raw_dimensions(dataset_path):
    """The viewer's dimensions for ``dataset_path``: the axes and voxel size
    (nm) of the finest level of the pyramid it belongs to -- the group, one
    of its levels or a precomputed volume (``wa.open_pyramid``), the array
    itself when it is in none -- x, y, z (viewer_space).

    Set before any layer is added, so that the raw data decides the viewer's
    coordinate space rather than whichever layer neuroglancer takes it from,
    such as an extra layer at another voxel size. None if it cannot be read.
    """
    try:
        pyramid = wa.open_pyramid(dataset_path, units="nanometer", strict_units=False)
        name, voxel_size = min(pyramid.voxel_sizes(), key=lambda level: tuple(level[1]))
        # The names the raw layer's volume gets: z, y, x when the store names none.
        meta = pyramid.level_meta(name)
        return viewer_space([meta.axes[i] for i in meta.spatial], voxel_size)
    except Exception as e:
        logger.warning(f"Could not read the viewer's dimensions from {dataset_path}: {e}")
        return None


def new_viewer(dataset_path, *, scales=None, raw=None, raw_name="data", layers=None):
    """A viewer showing the raw data at ``dataset_path``, then ``layers``.

    - ``scales``: the viewer's voxel size, z, y, x in nm. None takes it from
      the raw (raw_dimensions), or leaves the dimensions to neuroglancer if
      it cannot be read. Either way the dimensions are x, y, z (viewer_space).
    - ``raw``: the raw data's layer, if the caller built it (raw_layer);
      shown as ``raw_name``.
    - ``layers``: ``{name: layer}``, added after the raw in their order.

    Its server binds every interface: the browser is rarely on the machine
    the viewer runs on.
    """
    if scales is None:
        dimensions = raw_dimensions(dataset_path)
    else:
        dimensions = viewer_space(["z", "y", "x"], scales)
    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.gpu_memory_limit = GPU_MEMORY_LIMIT
        if dimensions is not None:
            s.dimensions = dimensions
        s.layers[raw_name] = raw_layer(dataset_path) if raw is None else raw
        for name, layer in (layers or {}).items():
            s.layers[name] = layer
    return viewer


def viewer_url(viewer):
    """The address to give a browser for ``viewer``: neuroglancer's own, with
    this machine's IP address (this_machine_ip) in place of its name.

    Neuroglancer names a server that binds every interface, as new_viewer's
    does, by the machine's DNS name. Off campus, a laptop's name still
    pointed at its campus address, and the viewer never loaded. A server
    named by an address or as localhost is bound to that alone, and is left
    as it is, as is anything without a host.
    """
    url = str(viewer)
    parsed = urlparse(url)
    host = parsed.hostname
    if not host or host == "localhost" or _is_address(host):
        return url
    ip = this_machine_ip()
    return parsed._replace(netloc=ip if parsed.port is None else f"{ip}:{parsed.port}").geturl()


def _is_address(host):
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True
