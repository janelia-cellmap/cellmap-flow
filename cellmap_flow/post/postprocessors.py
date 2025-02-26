import logging
import time
import numpy as np
import inspect
from cellmap_flow.utils.web_utils import encode_to_str, decode_to_json
import ast
import neuroglancer

logger = logging.getLogger(__name__)


class PostProcessor:

    @classmethod
    def name(cls):
        return cls.__name__

    def __call__(self, data: np.ndarray, chunk_corner=None) -> np.ndarray:
        return self.process(data, chunk_corner)

    def process(self, data, chunk_corner) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.dtype.kind in {"U", "O"}:
            try:
                data = data.astype(self.dtype)
            except ValueError:
                raise TypeError(
                    f"Cannot convert non-numeric data to float. Found dtype: {data.dtype}"
                )

        sig = inspect.signature(self._process)
        if "chunk_corner" in sig.parameters:
            data = self._process(data, chunk_corner)
        else:
            data = self._process(data)

        return data.astype(self.dtype)

    def _process(self, data, chunk_corner=None):
        raise NotImplementedError("Subclasses must implement this method")

    def to_dict(self):
        result = {"name": self.name()}
        for k, v in self.__dict__.items():
            result[k] = v
        return result

    @property
    def dtype(self):
        return np.uint8

    @property
    def is_segmentation(self):
        return False


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

    def to_dict(self):
        return {"name": self.name()}

    @property
    def dtype(self):
        return np.uint8


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

    def _process(self, data):
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


import mwatershed as mws
from scipy.ndimage import measurements
import fastremap


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
        self.bias = float(bias)
        self.neighborhood = ast.literal_eval(neighborhood)

    def _process(self, data):
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

        data = data.astype(np.uint64)
        fastremap.mask_except(segmentation, filtered_fragments, in_place=True)
        data[0] = segmentation
        # filtered_fragments = np.array(filtered_fragments, dtype=segmentation.dtype)
        # data[self.channel] = to_process
        return data

    def to_dict(self):
        return {"name": self.name()}

    @property
    def dtype(self):
        return np.uint64

    @property
    def is_segmentation(self):
        return True


class SimpleBlockwiseMerger(PostProcessor):
    def __init__(
        self,
    ):
        self.equivalences = neuroglancer.equivalence_map.EquivalenceMap()
        self.edge_voxel_position_to_id_dict = {}

    def _process(self, data, chunk_corner):
        mask = np.zeros_like(data[0], dtype=bool)
        mask[1:-1, 1:-1, 1:-1] = True
        data_masked = np.ma.masked_array(data[0], mask)
        z, y, x = np.ma.where(data_masked > 0)
        segmented_ids = data_masked[z, y, x]
        # current_edge_voxel_position_to_id_dict = dict(
        #     zip(
        #         zip(
        #             z + chunk_corner[0],
        #             y + chunk_corner[1],
        #             x + chunk_corner[2],
        #         ),
        #         segmented_ids,
        #     )
        # )
        t = time.time()
        before_len = len(self.edge_voxel_position_to_id_dict)
        self.edge_voxel_position_to_id_dict.update(
            dict(
                zip(
                    zip(
                        z + chunk_corner[0],
                        y + chunk_corner[1],
                        x + chunk_corner[2],
                    ),
                    segmented_ids,
                )
            )
        )
        t1 = time.time()
        after_len = len(self.edge_voxel_position_to_id_dict)
        self.calculate_equivalences()
        t2 = time.time()
        raise Exception(t1 - t, t2 - t1, before_len, after_len)
        # print(f"Edge voxel position to id dict: {self.edge_voxel_position_to_id_dict}")
        return data

    def to_dict(self):
        return {"name": self.name()}

    def calculate_equivalences(self):
        # copy because otherwise it can change sizes in loop since postprocessing may begin elsewhere
        edge_voxel_position_to_id_dict = self.edge_voxel_position_to_id_dict.copy()
        # Neighboring offsets (±1 in x, y, or z)
        offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for position, id in edge_voxel_position_to_id_dict.items():
            for offset in offsets:
                neighbor = (
                    position[0] + offset[0],
                    position[1] + offset[1],
                    position[2] + offset[2],
                )
                if (
                    neighbor_id := edge_voxel_position_to_id_dict.get(neighbor)
                ) not in (None, id):
                    self.equivalences.union(
                        id,
                        neighbor_id,
                    )

    @property
    def dtype(self):
        return np.uint64

    @property
    def is_segmentation(self):
        return True


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
    postoricessors = []
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
        postoricessors.append(
            {
                "class_name": post_cls.__name__,
                "name": post_name,
                "params": params,
            }
        )
    return postoricessors


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

# # %%
# import fastremap
# import numpy as np


# a = np.array([1, 2, 3])
# component_map = fastremap.component_map([1], [0])
# fastremap.mask_except(a, [1])
# # # %%
# # %%
# class Temp:
#     def __init__(self):
#         self.list = []

#     def __call__(self):
#         print("blah")
#         return self.process()

#     def process(self):
#         print("yo")
#         self.list.append(4)

# o =Temp()
# o()
# # %%
# o()
# o.list
# %%
# create large random dictionary
import numpy as np

# Parameters
num_entries = 1_000_000  # Number of key-value pairs

# Generate random coordinates and segmented IDs
x = np.random.randint(0, 1000, num_entries, dtype=np.uint64)
y = np.random.randint(0, 1000, num_entries, dtype=np.uint64)
z = np.random.randint(0, 1000, num_entries, dtype=np.uint64)
segmented_ids = np.random.randint(1, 10000, num_entries).astype(np.uint64)

# Create the first large dictionary
dict1 = {(z[i], y[i], x[i]): segmented_ids[i] for i in range(num_entries)}

# Generate another set for the second dictionary
x2 = np.random.randint(0, 1000, num_entries, dtype=np.uint64)
y2 = np.random.randint(0, 1000, num_entries, dtype=np.uint64)
z2 = np.random.randint(0, 1000, num_entries, dtype=np.uint64)
segmented_ids2 = np.random.randint(1, 10000, num_entries).astype(np.uint64)


# # Create the second large dictionary
dict2 = {(z2[i], y2[i], x2[i]): segmented_ids2[i] for i in range(num_entries)}
# # %%
# # %%
import neuroglancer
def calculate_it():
    equivalences = neuroglancer.equivalence_map.EquivalenceMap()
    offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    edge_voxel_position_to_id_dict = dict1.copy()
    precomputed_neighbors = {
        position: [
            (position[0] + dx, position[1] + dy, position[2] + dz) for dx, dy, dz in offsets
        ]
        for position in edge_voxel_position_to_id_dict
    }
    local_get = edge_voxel_position_to_id_dict.get
    for position, id in edge_voxel_position_to_id_dict.items():
        neighbors = precomputed_neighbors[position]
        # for neighbor in neighbors:
        #     if (neighbor_id := local_get(neighbor)) not in (
        #         None,
        #         id,
        #     ):
        #         equivalences.union(
        #             id,
        #             neighbor_id,
        #         )
calculate_it()
# # %%
# dict1.update(dict2)
# %%
