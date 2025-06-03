import logging
import numpy as np
import inspect
import ast
import neuroglancer
import pymorton
import threading

postprocessing_lock = threading.Lock()

logger = logging.getLogger(__name__)


class PostProcessor:

    @classmethod
    def name(cls):
        return cls.__name__

    def __call__(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return self.process(data, **kwargs)

    def process(self, data, **kwargs) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.dtype.kind in {"U", "O"}:
            try:
                data = data.astype(self.dtype)
            except ValueError:
                raise TypeError(
                    f"Cannot convert non-numeric data to float. Found dtype: {data.dtype}"
                )
        # if there are kwargs
        sig = inspect.signature(self._process)
        [kwargs.pop(k) for k in list(kwargs.keys()) if k not in sig.parameters]
        data = self._process(data, **kwargs)
        return data.astype(self.dtype)

    def _process(self, data, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def to_dict(self):
        result = {"name": self.name()}
        for k, v in self.__dict__.items():
            result[k] = v
        return result

    @property
    def dtype(self):
        return None

    @property
    def is_segmentation(self):
        return None


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

    def to_dict(self):
        return {"name": self.name(), "threshold": self.threshold}

    @property
    def dtype(self):
        return np.uint8

    @property
    def is_segmentation(self):
        return True


from scipy.ndimage import label


class LabelPostprocessor(PostProcessor):
    def __init__(self, channel: int = 0):
        self.channel = int(channel)

    def _process(self, data, chunk_corner, chunk_num_voxels):
        to_process = data[self.channel]
        to_process, num_features = label(to_process)
        data[self.channel] = to_process
        return data

    def to_dict(self):
        return {"name": self.name()}

    @property
    def dtype(self):
        return np.uint8

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
        #        if self.use_exact:
        morton_order_number = pymorton.interleave(*chunk_corner)
        unique_increment = chunk_num_voxels * morton_order_number
        if not self.use_exact:
            mixed = (unique_increment * 2654435761) & 0xFFFFFFFF
            mixed ^= mixed >> 16
            unique_increment = mixed & 0xFFFF
            # with postprocessing_lock:
            # unique_increment = self.num_previous_segments
            # self.num_previous_segments += len(
            #     fastremap.unique(to_process[to_process > 0])
            # )

        to_process[to_process > 0] += unique_increment.astype(to_process.dtype)
        data[self.channel] = to_process
        return data

    def to_dict(self):
        return {"name": self.name()}

    @property
    def dtype(self):
        return np.uint64 if self.use_exact else np.uint16

    @property
    def is_segmentation(self):
        return True


import mwatershed as mws
from scipy.ndimage import measurements
import fastremap
from funlib.math import cantor_number


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
        data = data / 255.0
        n_channels = data.shape[0]
        self.neighborhood = self.neighborhood[:n_channels]
        # raise Exception(data.max(), data.min(), self.neighborhood)

        segmentation = mws.agglom(
            data.astype(np.float64) - self.bias,
            self.neighborhood,
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
            # https://chatgpt.com/c/67c5db69-a3cc-8001-8be5-21d00cef0a8f
            # mixed = (unique_increment * 2654435761) & 0xFFFFFFFF
            # mixed ^= mixed >> 16
            # unique_increment = mixed & 0xFFFF  # with postprocessing_lock:
            # unique_increment = self.num_previous_segments
            # self.num_previous_segments += len(filtered_fragments)

        segmentation[segmentation > 0] += unique_increment
        segmentation = segmentation.astype(np.uint64 if self.use_exact else np.uint16)
        # for exact ids need the following: chunk_num_voxels * pymorton or funlib.math.cantor_number(chunk_corner), or pymorton?

        # filtered_fragments = np.array(filtered_fragments, dtype=segmentation.dtype)
        # data[self.channel] = to_process
        # insert empty dimension
        return np.expand_dims(segmentation, axis=0)

    def to_dict(self):
        return {"name": self.name()}

    @property
    def dtype(self):
        return np.uint64 if self.use_exact else np.uint16

    @property
    def is_segmentation(self):
        return True

    @property
    def num_channels(self):
        return 1


import fastmorph


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

    def _process(self, data, chunk_corner):
        segmentation = data[self.channel]
        for slice_reference, slice in self.slices.items():
            slice_data = segmentation[slice]
            if self.face_erosion_iterations > 0:
                slice_data = fastmorph.erode(
                    slice_data, iterations=self.face_erosion_iterations
                )
            coord_0, coord_1 = np.where(slice_data > 0)
            segmented_ids = slice_data[coord_0, coord_1]
            self.chunk_slice_position_to_coords_id_dict[
                (chunk_corner, slice_reference)
            ] = dict(
                zip(
                    zip(coord_0, coord_1),
                    segmented_ids,
                )
            )
        for key in self.keys_to_skip:
            self.chunk_slice_position_to_coords_id_dict.pop(key, None)
        self.calculate_equivalences()
        # print(f"Edge voxel position to id dict: {self.edge_voxel_position_to_id_dict}")
        return data.astype(np.uint64 if self.use_exact else np.uint16)

    def to_dict(self):
        return {"name": self.name()}

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
        self.channels = [int(channel) for channel in channels.split(",")]

    def _process(self, data):
        data = data[self.channels, :, :, :]
        return data

    def to_dict(self):
        return {"name": self.name()}

    @property
    def num_channels(self):
        return len(self.channels)


class LambdaPostprocessor(PostProcessor):
    def __init__(self, expression: str):
        self.expression = expression
        self._lambda = eval(f"lambda x: {expression}")

    def _process(self, data) -> np.ndarray:
        return self._lambda(data.astype(np.float32))

    def to_dict(self):
        return {"name": self.name(), "expression": self.expression}

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


def get_postprocessors(elms: dict) -> PostProcessor:
    result = []
    for post_name in elms:
        found = False
        for nm in PostProcessorMethods:
            if nm.name() == post_name:
                result.append(nm(**elms[post_name]))
                found = True
                break
        if not found:
            raise ValueError(f"PostProcess method {post_name} not found")
    return result


PostProcessorMethods = [f for f in PostProcessor.__subclasses__()]
