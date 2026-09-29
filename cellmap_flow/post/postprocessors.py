import logging
import numpy as np
import inspect
import ast
import neuroglancer
import pymorton
import threading
from scipy.ndimage import label
import mwatershed as mws
from scipy.ndimage import measurements
import fastremap
import fastmorph
from cellmap_flow.norm.input_normalize import SerializableInterface, deserialize_list
from cellmap_flow.utils.safe_expression import compile_expression

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


class LabelPostprocessor(PostProcessor):
    def __init__(self, channel: int = 0):
        self.channel = int(channel)

    def _process(self, data, chunk_corner, chunk_num_voxels):
        # Into a new uint32 array: writing the labels back into the model's
        # own (often uint8) array wrapped every id above 255, and the declared
        # uint8 dtype wrapped them again on the way out.
        labels, _ = label(data[self.channel])
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
        data = data.astype(np.uint64 if self.use_exact else np.uint16)
        to_process = data[self.channel]
        morton_order_number = pymorton.interleave(*chunk_corner)
        unique_increment = chunk_num_voxels * morton_order_number
        if not self.use_exact:
            mixed = (unique_increment * 2654435761) & 0xFFFFFFFF
            mixed ^= mixed >> 16
            unique_increment = mixed & 0xFFFF

        to_process[to_process > 0] += unique_increment.astype(to_process.dtype)
        data[self.channel] = to_process
        return data

    @property
    def dtype(self):
        return np.uint64 if self.use_exact else np.uint16

    @property
    def is_segmentation(self):
        return True


class AffinityPostprocessor(PostProcessor):
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
        # Integer input is the 0-255 that DefaultPostprocessor produces (the
        # usual chain), so scale it back to [0, 1] exactly as before. Float
        # input is already an affinity in [0, 1] (e.g. straight after a
        # SigmoidPostprocessor); dividing that by 255 as well left every edge
        # near zero and the watershed merged everything.
        if np.issubdtype(data.dtype, np.integer) or data.dtype == np.bool_:
            data = data / 255.0
        else:
            data = data.astype(np.float64)
        n_channels = data.shape[0]
        # Local, not self.neighborhood: truncating the attribute made every
        # later call use the first chunk's channel count.
        neighborhood = self.neighborhood[:n_channels]

        segmentation = mws.agglom(
            data.astype(np.float64) - self.bias,
            neighborhood,
        )

        # filter fragments
        average_affs = np.mean(data, axis=0)

        filtered_fragments = []

        fragment_ids = fastremap.unique(segmentation[segmentation > 0])

        for fragment, mean in zip(
            fragment_ids, measurements.mean(average_affs, segmentation, fragment_ids)
        ):
            if mean >= self.bias:
                filtered_fragments.append(fragment)

        fastremap.mask_except(segmentation, filtered_fragments, in_place=True)
        fastremap.renumber(segmentation, in_place=True)
        unique_increment = chunk_num_voxels * pymorton.interleave(*chunk_corner)
        if not self.use_exact:
            unique_increment = np.random.randint(0, 256) * 256

        # numpy has no common integer type for uint64 and int64, so
        # ``np.result_type(np.uint64, np.int64)`` is float64 -- an in-place add of a
        # numpy *signed* scalar into a uint64 array therefore raises
        # UFuncOutputCastingError. Both increments above are numpy int64
        # (np.prod / np.random.randint), so cast explicitly to keep the add in
        # uint64. (A plain Python int would also work under NEP 50's weak
        # promotion, which is why this never reproduced with literal values.)
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
