# The segmentation libraries (and neuroglancer, scipy.ndimage) are imported
# inside the steps that use them: importing this module is how every chain
# is read, including in processes that never run a segmentation step, and
# together they took seconds to load. The instance segmenters themselves
# live in post.segment, shared with the Finetune tab's seeding, which
# imports them the same way.
import ast
import inspect
import logging
import threading

import numpy as np

from cellmap_flow.norm.input_normalize import SerializableInterface, deserialize_list
from cellmap_flow.norm.safe_expression import compile_expression
from cellmap_flow.post import segment

logger = logging.getLogger(__name__)


class PostProcessor(SerializableInterface):
    """Base class for post-processing methods."""

    def __init__(self):
        # Explicit empty __init__ so the GUI doesn't introspect *args/**kwargs
        # for subclasses that don't define their own signature.
        pass

    @property
    def is_segmentation(self):
        return None

    def problem_with(self, model_config):
        """Why this step cannot run on ``model_config``'s output, or None when it can.

        Checked by the dashboard before it applies a chain: a step that
        fails on every chunk does so inside the model's server, where the
        page shows only an empty layer.
        """
        return None


class SigmoidPostprocessor(PostProcessor):
    """Apply sigmoid activation to convert logits to probabilities."""

    def _process(self, data):
        return 1.0 / (1.0 + np.exp(-data.astype(np.float32)))

    @property
    def dtype(self):
        return np.float32


class DefaultPostprocessor(PostProcessor):
    def __init__(
        self,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        bias: float = 1.0,
        multiplier: float = 127.5,
    ):
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.bias = float(bias)
        self.multiplier = float(multiplier)

    def _process(self, data):
        data = data.clip(self.clip_min, self.clip_max)
        data = (data + self.bias) * self.multiplier
        return data.astype(np.uint8)

    @property
    def dtype(self):
        return np.uint8

    @property
    def is_segmentation(self):
        return False


class ThresholdPostprocessor(PostProcessor):
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def _process(self, data):
        data = (data.astype(np.float32) > self.threshold).astype(np.uint8)
        return data

    @property
    def dtype(self):
        return np.uint8

    @property
    def is_segmentation(self):
        return True


class FillHolesPostprocessor(PostProcessor):
    """Threshold, then fill the background holes each foreground blob encloses.

    For compact single-instance organelles (a nucleus, say) that never have
    interior gaps; not for structures with a real lumen. It runs per chunk
    with no halo, so a hole that touches the chunk's boundary is not enclosed
    within the chunk and stays unfilled.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def _process(self, data):
        import fastmorph

        binary = data.astype(np.float32) > self.threshold
        if binary.ndim == 3:
            filled = fastmorph.fill_holes(binary, remove_enclosed=True)
        elif binary.ndim == 4:
            # fastmorph.fill_holes takes at most 3-D input.
            filled = np.stack(
                [fastmorph.fill_holes(channel, remove_enclosed=True) for channel in binary]
            )
        else:
            raise ValueError(
                f"FillHolesPostprocessor expects (z, y, x) or (c, z, y, x) data, "
                f"got shape {data.shape}"
            )
        return filled.astype(np.uint8)

    @property
    def dtype(self):
        return np.uint8

    @property
    def is_segmentation(self):
        return True


class LabelPostprocessor(PostProcessor):
    """An id per connected object of one channel's nonzero voxels (``segment.connected_components``).

    ``connectivity``: 1 faces (the default), 2 faces and edges, 3 all
    neighbours. ``min_size``: objects of fewer voxels become background.
    ``per_slice``: each z slice labelled on its own, in 2D. The other
    channels pass through, as uint32.
    """

    def __init__(self, channel: int = 0, connectivity: int = 1, min_size: int = 0, per_slice: bool = False):
        self.channel = int(channel)
        self.connectivity = segment.as_connectivity(connectivity)
        self.min_size = int(min_size)
        self.per_slice = segment.as_bool(per_slice)

    def _process(self, data, chunk_corner, chunk_num_voxels):
        # Into a new uint32 array: writing the labels back into the model's
        # own (often uint8) array wrapped every id above 255, and the declared
        # uint8 dtype wrapped them again on the way out.
        labels = segment.connected_components(
            data[self.channel], self.connectivity, self.min_size, self.per_slice
        )
        out = data.astype(np.uint32)
        out[self.channel] = labels
        return out

    @property
    def dtype(self):
        return np.uint32

    @property
    def is_segmentation(self):
        return True


class MortonSegmentationRelabeling(PostProcessor):
    def __init__(self, channel: int = 0):
        use_exact = "True"
        self.channel = int(channel)
        self.num_previous_segments = 0
        self.use_exact = use_exact == "True"

    def _process(self, data, chunk_corner, chunk_num_voxels):
        import pymorton

        data = data.astype(np.uint64 if self.use_exact else np.uint16)
        to_process = data[self.channel]
        # A Python int, whatever the caller passes (the Inferencer passes a
        # plain int), so the product cannot overflow before it is cast.
        unique_increment = int(chunk_num_voxels) * pymorton.interleave(*chunk_corner)
        if not self.use_exact:
            mixed = (unique_increment * 2654435761) & 0xFFFFFFFF
            mixed ^= mixed >> 16
            unique_increment = mixed & 0xFFFF

        to_process[to_process > 0] += to_process.dtype.type(unique_increment)
        data[self.channel] = to_process
        return data

    @property
    def dtype(self):
        return np.uint64 if self.use_exact else np.uint16

    @property
    def is_segmentation(self):
        return True


class AffinityPostprocessor(PostProcessor):
    """Objects from affinities by mutex watershed (``segment.mutex_watershed``).

    ``bias``: the affinity above which neighbours join, and below which a
    fragment's mean makes it background. ``neighborhood``: the offset each
    channel compares, as the model was trained with. Ids are offset by the
    chunk's Morton index, so they are unique across chunks.
    """

    def __init__(
        self,
        bias: float = 0.0,
        neighborhood: str = """[
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [3, 0, 0],
                [0, 3, 0],
                [0, 0, 3],
                [9, 0, 0],
                [0, 9, 0],
                [0, 0, 9],
            ]""",
    ):
        use_exact = "True"
        self.bias = float(bias)
        self.neighborhood = ast.literal_eval(neighborhood)
        self.use_exact = use_exact == "True"
        self.num_previous_segments = 0

    def _process(self, data, chunk_num_voxels, chunk_corner):
        import pymorton

        # Integer input is the 0-255 that DefaultPostprocessor produces (the
        # usual chain), so scale it back to [0, 1] exactly as before. Float
        # input is already an affinity in [0, 1] (e.g. straight after a
        # SigmoidPostprocessor); dividing that by 255 as well left every edge
        # near zero and the watershed merged everything.
        if np.issubdtype(data.dtype, np.integer) or data.dtype == np.bool_:
            data = data / 255.0
        # Cut to the channels inside the call, not on self.neighborhood:
        # truncating the attribute made every later call use the first
        # chunk's channel count.
        segmentation = segment.mutex_watershed(data, self.neighborhood, self.bias)
        unique_increment = chunk_num_voxels * pymorton.interleave(*chunk_corner)
        if not self.use_exact:
            unique_increment = np.random.randint(0, 256) * 256

        # numpy has no common integer type for uint64 and int64, so
        # ``np.result_type(np.uint64, np.int64)`` is float64 -- an in-place add of a
        # numpy *signed* scalar into a uint64 array therefore raises
        # UFuncOutputCastingError. np.random.randint gives a numpy int64, and a
        # caller may pass chunk_num_voxels as one, so cast explicitly to keep
        # the add in uint64. (A plain Python int would also work under NEP 50's
        # weak promotion, which is why this never reproduced with literal values.)
        segmentation[segmentation > 0] += np.uint64(unique_increment)
        segmentation = segmentation.astype(np.uint64 if self.use_exact else np.uint16)
        # insert empty dimension
        return np.expand_dims(segmentation, axis=0)

    @property
    def dtype(self):
        return np.uint64 if self.use_exact else np.uint16

    @property
    def is_segmentation(self):
        return True

    @property
    def num_channels(self):
        return 1


class SimpleBlockwiseMerger(PostProcessor):
    # NOTE: Need to be careful since this can be called in parallel and some things may change size during loops etc.
    def __init__(
        self,
        channel: int = 0,
        face_erosion_iterations: int = 0,
    ):
        import neuroglancer

        use_exact = "True"
        self.channel = int(channel)
        self.face_erosion_iterations = int(face_erosion_iterations)
        self.use_exact = use_exact == "True"
        self.equivalences = neuroglancer.equivalence_map.EquivalenceMap()
        self.chunk_slice_position_to_coords_id_dict = {}
        # -1: for start and 1 for end
        self.slices = {
            (-1, 0, 0): (0, slice(None), slice(None)),
            (1, 0, 0): (-1, slice(None), slice(None)),
            (0, -1, 0): (slice(None), 0, slice(None)),
            (0, 1, 0): (slice(None), -1, slice(None)),
            (0, 0, -1): (slice(None), slice(None), 0),
            (0, 0, 1): (slice(None), slice(None), -1),
        }
        self.keys_to_skip = set()
        # The server calls one instance from every Flask request thread; this
        # guards the dict, the set and the equivalence map they all share.
        self._lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def equivalences_json(self):
        """The equivalences so far, read while no other chunk is adding to them."""
        with self._lock:
            return self.equivalences.to_json()

    def _process(self, data, chunk_corner):
        import fastmorph

        segmentation = data[self.channel]
        faces = {}
        for slice_reference, slice in self.slices.items():
            slice_data = segmentation[slice]
            if self.face_erosion_iterations > 0:
                slice_data = fastmorph.erode(
                    slice_data, iterations=self.face_erosion_iterations
                )
            coord_0, coord_1 = np.where(slice_data > 0)
            segmented_ids = slice_data[coord_0, coord_1]
            faces[(chunk_corner, slice_reference)] = dict(
                zip(
                    zip(coord_0, coord_1),
                    segmented_ids,
                )
            )
        with self._lock:
            self.chunk_slice_position_to_coords_id_dict.update(faces)
            for key in self.keys_to_skip:
                self.chunk_slice_position_to_coords_id_dict.pop(key, None)
            self.calculate_equivalences()
        return data.astype(np.uint64 if self.use_exact else np.uint16)

    def calculate_equivalences(self):
        chunk_slice_position_to_coords_id_dict = (
            self.chunk_slice_position_to_coords_id_dict.copy()
        )
        for (
            current_slice_key,
            coords_id_dict1,
        ) in chunk_slice_position_to_coords_id_dict.items():
            if current_slice_key in self.keys_to_skip:
                continue
            chunk_corner = np.array(current_slice_key[0])
            slice = np.array(current_slice_key[1])
            neighboring_slice_key = (tuple(chunk_corner + slice), tuple(-1 * slice))
            if coords_id_dict2 := chunk_slice_position_to_coords_id_dict.get(
                neighboring_slice_key
            ):
                if neighboring_slice_key in self.keys_to_skip:
                    continue
                coords_id_dict2 = chunk_slice_position_to_coords_id_dict[
                    neighboring_slice_key
                ]
                for position, id1 in coords_id_dict1.items():
                    if id2 := coords_id_dict2.get(position):
                        self.equivalences.union(id1, id2)
                self.keys_to_skip.add(current_slice_key)
                self.keys_to_skip.add(neighboring_slice_key)

    @property
    def dtype(self):
        return np.uint64 if self.use_exact else np.uint16

    @property
    def is_segmentation(self):
        return True


class CellposeMasksPostprocessor(PostProcessor):
    """Cellpose's masks, made from a Cellpose model's flows output.

    Cellpose turns its three output channels into objects by following each
    pixel's flow to where the flows converge. Run here, on a server with
    ``output: flows`` (channels flow_y, flow_x and cell, the probability),
    its thresholds can be changed from the dashboard and the layer redrawn,
    where ``output: masks`` fixes them when the server starts. Put it first
    in the chain, on the flows as served. It needs Cellpose, which a
    Cellpose model's server has.

    Each slice is segmented on its own, as Cellpose segments a 2D image;
    ``stitch_threshold`` above 0 joins a slice's object to the next slice's
    one it overlaps by that IoU (Cellpose's stitch3D), else each slice's
    ids follow the previous slice's. Ids are unique within the chunk:
    follow with MortonSegmentationRelabeling for ids unique across chunks.

    Args:
        flow_threshold: how far an object's flows may be from those its
            shape implies before it is dropped (Cellpose's; 0 keeps all).
        cellprob_threshold: the cell probability, as a logit, over which a
            pixel is followed (Cellpose's; 0 is a probability of 0.5).
        min_size: objects of fewer pixels are removed (per slice; in 3D
            after stitching).
        stitch_threshold: IoU joining objects across slices; 0 is off.
        niter: flow-following steps (Cellpose's default, 200).
    """

    def __init__(self, flow_threshold: float = 0.4, cellprob_threshold: float = 0.0, min_size: int = 15,
                 stitch_threshold: float = 0.0, niter: int = 200):
        self.flow_threshold = float(flow_threshold)
        self.cellprob_threshold = float(cellprob_threshold)
        self.min_size = int(min_size)
        self.stitch_threshold = float(stitch_threshold)
        self.niter = int(niter)

    def problem_with(self, model_config):
        """None for a Cellpose model (or a finetune of one) served with ``output: flows``."""
        base = getattr(model_config, "base_model_config", None) or model_config
        name = getattr(model_config, "name", "the model")
        if getattr(type(base), "cli_name", None) != "cellpose":
            return f"{name} is not a Cellpose model: CellposeMasksPostprocessor needs Cellpose's flows"
        if getattr(base, "output", None) != "flows":
            return (f"{name} serves Cellpose's {getattr(base, 'output', '?')}, not its flows: run it with "
                    "Output: Flows for CellposeMasksPostprocessor (or Output: Masks for masks without it)")
        return None

    def _process(self, data):
        try:
            import torch
            from cellpose import dynamics, utils
        except ImportError as e:
            raise RuntimeError(
                "CellposeMasksPostprocessor needs Cellpose, which this server's environment does not have: "
                "use it on a Cellpose model's server (output: flows)"
            ) from e
        if data.ndim != 4 or data.shape[0] != 3:
            raise ValueError(
                "CellposeMasksPostprocessor needs a Cellpose flows output, 3 channels (flow_y, flow_x, cell), "
                f"first in the chain; got shape {data.shape}"
            )
        flows = np.asarray(data[:2], dtype=np.float32)
        probability = np.clip(np.asarray(data[2], dtype=np.float64), 1e-7, 1 - 1e-7)
        cellprob = np.log(probability / (1 - probability)).astype(np.float32)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        stitch = self.stitch_threshold > 0 and data.shape[1] > 1
        masks = np.zeros(data.shape[1:], dtype=np.uint32)
        offset = 0
        for z in range(data.shape[1]):
            slice_masks = dynamics.resize_and_compute_masks(
                flows[:, z], cellprob[z], niter=self.niter, cellprob_threshold=self.cellprob_threshold,
                flow_threshold=self.flow_threshold, min_size=-1 if stitch else self.min_size,
                max_size_fraction=0.4, device=device,
            ).astype(np.uint32)
            if not stitch:
                slice_masks[slice_masks > 0] += offset
                offset = max(offset, int(slice_masks.max()))
            masks[z] = slice_masks
        if stitch:
            masks = utils.stitch3D(masks, stitch_threshold=self.stitch_threshold).astype(np.uint32)
            if self.min_size > 0:
                masks = utils.fill_holes_and_remove_small_masks(masks, min_size=self.min_size).astype(np.uint32)
        return masks[np.newaxis]

    @property
    def dtype(self):
        return np.uint32

    @property
    def is_segmentation(self):
        return True


class ChannelSelection(PostProcessor):
    def __init__(self, channels: str = "0"):
        # "0,2" from the dashboard form; YAML may also give 2 or [0, 2].
        if isinstance(channels, str):
            channels = channels.split(",")
        elif not isinstance(channels, (list, tuple)):
            channels = [channels]
        self.channels = [int(channel) for channel in channels]

    def _process(self, data):
        data = data[self.channels, :, :, :]
        return data

    @property
    def num_channels(self):
        return len(self.channels)


class LambdaPostprocessor(PostProcessor):
    def __init__(self, expression: str):
        self.expression = expression
        self._lambda = compile_expression(expression)

    def _process(self, data) -> np.ndarray:
        return self._lambda(data.astype(np.float32))

    @property
    def dtype(self):
        return np.float32


def get_postprocessors_list() -> list[dict]:
    """Returns a list of dictionaries containing the names and parameters of all subclasses of PostProcessor."""
    postprocess_classes = PostProcessor.__subclasses__()
    postprocessors = []
    for post_cls in postprocess_classes:
        post_name = post_cls.__name__
        sig = inspect.signature(post_cls.__init__)
        params = {}
        for param_name, param_obj in sig.parameters.items():
            if param_name == "self":
                continue
            default_val = param_obj.default
            if default_val is inspect._empty:
                default_val = ""
            params[param_name] = default_val
        postprocessors.append(
            {
                "class_name": post_cls.__name__,
                "name": post_name,
                "params": params,
            }
        )
    return postprocessors


def get_postprocessors(elms) -> list[PostProcessor]:
    """Get postprocessors from either dict or list format."""
    return deserialize_list(elms, PostProcessor)
