"""A model's geometry: what it reads and writes, and at what voxel size.

The inference server, blockwise, the server's ``model_info`` and the
dashboard's geometry cache each read these numbers off ``ModelConfig.config``
one attribute at a time, with their own conversions. ``ModelGeometry`` is
them, read once: sizes in nm as given (whole numbers as ints, so 5.24 nm
stays 5.24), and the shapes in voxels, the context and the block shape
derived from them.

``from_config`` reads anything with the config attributes: a built config, or
the stand-ins ``serving.client`` and ``models.geometry_cache`` build from a
server's ``model_info`` or the cache, which carry no block shape.

Importing this module is cheap: numpy and funlib.geometry only.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from funlib.geometry import Coordinate

# The names a chunk_output_axes entry may give the channel axis.
CHANNEL_AXES = ("c", "c^", "channel")
DEFAULT_OUTPUT_AXES = ("c", "z", "y", "x")


def number(value):
    """``value`` as an int when it is whole, else as a float.

    ``int()`` truncated a 5.24 nm voxel size to 5, which put crops and
    layers on the wrong grid.
    """
    value = float(value)
    return int(value) if value.is_integer() else value


def _numbers(values) -> tuple:
    return tuple(number(v) for v in values)


def _voxels(extent, voxel_size) -> Tuple[int, ...]:
    # 52.4 // 5.24 is 9.0 in floats; a quotient within rounding of a whole
    # number is that number.
    quotient = np.asarray(extent, dtype=float) / np.asarray(voxel_size, dtype=float)
    nearest = np.rint(quotient)
    return tuple(int(v) for v in np.where(np.isclose(quotient, nearest), nearest, np.floor(quotient)))


def _channel_names(names) -> Optional[Tuple[str, ...]]:
    """``names`` as a tuple of strings, or None when it names no channel.

    A string is one name: iterating it would make every letter a channel.
    """
    if isinstance(names, str):
        names = [names] if names else []
    if names is None or len(names) == 0:
        return None
    return tuple(str(c) for c in names)


def channel_names_of(config):
    """The channel names a config gives, as a tuple, or None: the first
    non-empty one of channels, channels_names (Hugging Face), classes, or a
    ModelGeometry's channel_names. A string is one name, so
    ``channels = "mito"`` is ``("mito",)``."""
    for attr in ("channels", "channels_names", "classes", "channel_names"):
        names = _channel_names(getattr(config, attr, None))
        if names is not None:
            return names
    return None


@dataclass(frozen=True)
class ModelGeometry:
    """What a model reads and writes. Sizes are in nm unless named ``*_shape`` in voxels."""

    input_voxel_size: Tuple[float, ...]
    output_voxel_size: Tuple[float, ...]
    read_shape: Tuple[float, ...]  # nm
    write_shape: Tuple[float, ...]  # nm
    output_channels: int
    channel_names: Optional[Tuple[str, ...]] = None
    chunk_output_axes: Tuple[str, ...] = DEFAULT_OUTPUT_AXES
    output_dtype: Optional[np.dtype] = None
    input_channels: int = 1
    # A config's own block_shape, (*spatial, channels). Configs must declare
    # one, it is checked against the model's output as before, and it stays
    # what the server and blockwise chunk by; block_shape() derives one only
    # when there is none (a server's model_info, the cache).
    declared_block_shape: Optional[Tuple[int, ...]] = field(default=None, compare=False)

    def __post_init__(self):
        def set_(name, value):  # frozen: normalize once, here
            object.__setattr__(self, name, value)

        for name in ("input_voxel_size", "output_voxel_size", "read_shape", "write_shape"):
            set_(name, _numbers(getattr(self, name)))
        set_("output_channels", int(self.output_channels))
        set_("input_channels", int(self.input_channels))
        set_("chunk_output_axes", tuple(self.chunk_output_axes))
        set_("channel_names", _channel_names(self.channel_names))
        if self.output_dtype is not None:
            set_("output_dtype", np.dtype(self.output_dtype))
        if self.declared_block_shape is not None:
            set_("declared_block_shape", tuple(int(v) for v in self.declared_block_shape))

    @property
    def ndim(self) -> int:
        """Spatial dimensions."""
        return len(self.input_voxel_size)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Voxels the model reads."""
        return _voxels(self.read_shape, self.input_voxel_size)

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Voxels the model writes."""
        return _voxels(self.write_shape, self.output_voxel_size)

    @property
    def context(self) -> Coordinate:
        """nm read beyond the written region on each side.

        funlib arithmetic, as the Inferencer and blockwise always computed
        it: Coordinate truncates each size to an int, and ``/`` floors.
        """
        return (Coordinate(self.read_shape) - Coordinate(self.write_shape)) / 2

    @property
    def has_channel_axis(self) -> bool:
        """Whether a chunk the model outputs has a channel axis."""
        return any(axis in self.chunk_output_axes for axis in CHANNEL_AXES)

    def block_shape(self) -> Tuple[int, ...]:
        """(*spatial, channels) of one output block: the declared one, if any."""
        if self.declared_block_shape is not None:
            return self.declared_block_shape
        return (*self.output_shape, self.output_channels)

    @classmethod
    def from_config(cls, config, chunk_output_axes=None, output_dtype=None) -> "ModelGeometry":
        """From ``ModelConfig.config``, or anything with its geometry attributes.

        ``chunk_output_axes`` and ``output_dtype``, when given, win over the
        config's own (``ModelConfig`` passes its properties, with their
        defaults); otherwise the config's are used, if it has them.
        """
        if chunk_output_axes is None:
            chunk_output_axes = getattr(config, "chunk_output_axes", DEFAULT_OUTPUT_AXES)
        if output_dtype is None:
            output_dtype = getattr(config, "output_dtype", None)
        return cls(
            input_voxel_size=config.input_voxel_size,
            output_voxel_size=config.output_voxel_size,
            read_shape=config.read_shape,
            write_shape=config.write_shape,
            output_channels=config.output_channels,
            channel_names=channel_names_of(config),
            chunk_output_axes=chunk_output_axes,
            output_dtype=output_dtype,
            input_channels=getattr(config, "input_channels", 1),
            declared_block_shape=getattr(config, "block_shape", None),
        )

    def to_model_info(self, spatial_axes=None, actual_input_voxel_size=None) -> dict:
        """The geometry keys of the server's ``/__control__/model_info``.

        ``spatial_axes``: the served array's, which are the raw data's (the
        spatial axes of chunk_output_axes by default). ``output_axes`` is
        them in order, then "c" when the output has a channel axis.

        ``actual_input_voxel_size``: what the input really is. When it is
        not input_voxel_size, the data was read at another level as if it
        were at input_voxel_size, voxel for voxel, so an output voxel really
        is ``actual_in * output_voxel_size / input_voxel_size``: that is
        ``effective_output_voxel_size``, otherwise output_voxel_size. The
        served .zattrs still say output_voxel_size.
        """
        if spatial_axes is None:
            spatial_axes = [a for a in self.chunk_output_axes if a not in CHANNEL_AXES]
        effective = self.output_voxel_size
        if actual_input_voxel_size is not None:
            actual = np.asarray(actual_input_voxel_size, dtype=float)
            declared = np.asarray(self.input_voxel_size, dtype=float)
            if not np.allclose(actual, declared):
                effective = _numbers(actual * np.asarray(effective, dtype=float) / declared)
        return {
            "output_channels": self.output_channels,
            "channels": list(self.channel_names) if self.channel_names else None,
            "write_shape": list(self.write_shape),
            "read_shape": list(self.read_shape),
            "output_voxel_size": list(self.output_voxel_size),
            "input_voxel_size": list(self.input_voxel_size),
            "effective_output_voxel_size": list(effective),
            "has_channel": self.has_channel_axis,
            "output_axes": list(spatial_axes) + (["c"] if self.has_channel_axis else []),
        }
