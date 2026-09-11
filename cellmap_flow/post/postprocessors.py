import logging
import numpy as np
import inspect
import ast
import neuroglancer
import pymorton
import threading
from scipy.ndimage import label
from skimage.segmentation import watershed
from scipy.spatial import cKDTree
import mwatershed as mws
from scipy.ndimage import measurements
import fastremap
from funlib.math import cantor_number
import fastmorph
from cellmap_flow.norm.input_normalize import SerializableInterface, deserialize_list

postprocessing_lock = threading.Lock()

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
        to_process = data[self.channel]
        to_process, num_features = label(to_process)
        data[self.channel] = to_process
        return data

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

    # def to_dict(self):
    #     return {"name": self.name()}

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

    # def to_dict(self):
    #     return {"name": self.name()}

    @property
    def dtype(self):
        return np.uint64 if self.use_exact else np.uint16

    @property
    def is_segmentation(self):
        return True

    @property
    def num_channels(self):
        return 1


class SkootsPostprocessor(PostProcessor):
    """Turn an already-decoded 5-channel SKOOTS head (semantic, skeleton,
    vec_z, vec_y, vec_x -- channels 0-1 true [0,1] probabilities, channels
    2-4 displacement in voxels; see `SkootsDecodeWrapper` in
    my_yamls/jrc_axolotl-heart-1_mito_skoots_finetuned.py, which applies
    that decode in the model's forward pass so thresholds here are on
    meaningful units instead of raw logits/unbounded regression) into an
    instance segmentation for a single block.

    Algorithm, per foreground connected-component in this block:
      1. threshold semantic/skeleton channels into binary masks.
      2. label the skeleton mask *restricted to that component* -- each
         skeleton connected-component is one candidate instance seed.
      3. if the component has no skeleton seed at all (its true skeleton
         lies entirely in a neighboring block), keep it as one provisional
         instance -- chain `SimpleBlockwiseMerger` afterward to stitch it
         to whatever id the neighboring block assigns to the touching
         piece; the merger only needs matching face voxels, not matching
         algorithms, so it works unchanged for this postprocessor too.
      4. otherwise, decode each foreground voxel's vector to an "embedded"
         position (voxel coord + predicted displacement, which the training
         target points *toward* the voxel's own instance's nearest skeleton
         point) and assign it the id of whichever of *this component's*
         skeleton seeds has the closest skeleton voxel to that embedded
         position. This is what actually splits two touching/close
         instances that share one semantic blob -- gating candidates to the
         component the voxel is already in keeps a stray vector from
         grabbing a seed id from a spatially distant, unrelated object.

    Restricting to per-component skeleton labeling (rather than one
    skeleton-mask connected-components pass over the whole block) also
    protects against a skeleton mask that's slightly too thick and bridges
    two objects' skeletons across a gap that isn't itself foreground --
    with per-component gating those two skeleton pieces can still only earn
    non-foreground gap voxels below (they're outside the component/foreground
    mask entirely), it does not by itself fix a skeleton mask that bridges
    *within* one continuous foreground blob (that's a training-target
    thickness problem, see skoots_targets.py::skeleton_mask_target).

    IDs are local-unique within the block, then offset by `chunk_corner`
    (same trick as `AffinityPostprocessor`) so they stay globally unique
    across blocks without a merge step. This produces *some* consistent
    per-block labeling, not a final cross-block-stitched segmentation --
    see class docstring precedent (`AffinityPostprocessor`/
    `SimpleBlockwiseMerger`) for why: blockwise inference here (daisy,
    `cellmap_flow/blockwise/blockwise_processor.py`) never gives a
    postprocessor access to neighboring blocks' data, only this block's own
    write_roi-sized model output. `SimpleBlockwiseMerger` can stitch
    instances across block faces for a single long-running interactive
    viewer/server process (it works by duck-typed `equivalences` polling in
    `server.py`, not by touching the written data) -- chain it right after
    this class in the yaml's `postprocess:` list for that. It does *not*
    work for the separate multi-worker `cellmap_flow_blockwise` batch export
    path (each bsub worker is an independent process with its own empty
    equivalence map) -- a real cross-block stitching pass for that path
    doesn't exist yet anywhere in this codebase and would need to be added
    separately (e.g. an offline union-find over block-boundary faces,
    applied as a final relabeling pass over the written zarr).
    """

    def __init__(
        self,
        semantic_channel: int = 0,
        skeleton_channel: int = 1,
        vector_channels: str = "[2, 3, 4]",
        semantic_threshold: float = 0.5,
        skeleton_threshold: float = 0.5,
        min_skeleton_size: int = 0,
    ):
        # Every param here must survive a str(stored_value) -> re-parse
        # round trip: `refresh_dataset` (server.py) rebuilds a fresh
        # postprocessor instance from `to_dict()`'s stringified attrs on
        # every request (see serialize_norms_posts_to_json / decode_to_json
        # + deserialize_list). ast.literal_eval on a python-list-repr string
        # is idempotent under that round trip (matches AffinityPostprocessor's
        # `neighborhood`); `"2,3,4".split(",")` is not, once the parsed list
        # itself gets stringified back to `"[2, 3, 4]"` on the next request.
        self.semantic_channel = int(semantic_channel)
        self.skeleton_channel = int(skeleton_channel)
        self.vector_channels = ast.literal_eval(vector_channels)
        self.semantic_threshold = float(semantic_threshold)
        self.skeleton_threshold = float(skeleton_threshold)
        # Dropping a low skeleton_threshold way down (e.g. to catch faint
        # true skeleton) also lets through isolated speckle -- a handful of
        # stray voxels barely over threshold with no real skeleton structure.
        # Each speckle still forms its own connected component and becomes a
        # seed, manufacturing a spurious tiny instance. Filtering candidate
        # skeleton components below this voxel count before they're used as
        # seeds removes that noise; their would-be seed voxels either fall to
        # a remaining real seed in the same foreground component, or (if none
        # remain) the whole component becomes provisional, same as the
        # existing num_local_skel == 0 path.
        self.min_skeleton_size = int(min_skeleton_size)

    def _process(self, data, chunk_corner, chunk_num_voxels):
        data = data.astype(np.float32)
        semantic_prob = data[self.semantic_channel]
        skeleton_prob = data[self.skeleton_channel]
        vectors = data[self.vector_channels]

        fg_mask = semantic_prob > self.semantic_threshold
        skel_mask = (skeleton_prob > self.skeleton_threshold) & fg_mask

        fg_components, num_fg = label(fg_mask)
        segmentation = np.zeros(fg_mask.shape, dtype=np.int64)
        next_id = 1

        for comp_id in range(1, num_fg + 1):
            comp_mask = fg_components == comp_id
            coords = np.argwhere(comp_mask)

            local_skel_labels, num_local_skel = label(comp_mask & skel_mask)
            if self.min_skeleton_size > 0 and num_local_skel > 0:
                sizes = np.bincount(local_skel_labels.ravel())
                too_small = np.nonzero(sizes < self.min_skeleton_size)[0]
                too_small = too_small[too_small != 0]
                if too_small.size:
                    local_skel_labels[np.isin(local_skel_labels, too_small)] = 0
                    local_skel_labels, num_local_skel = label(local_skel_labels > 0)

            if num_local_skel == 0:
                # true skeleton not visible in this block -- provisional id,
                # left for SimpleBlockwiseMerger (or a future offline
                # stitching pass) to reconcile with a neighboring block.
                segmentation[coords[:, 0], coords[:, 1], coords[:, 2]] = next_id
                next_id += 1
                continue

            skel_coords = np.argwhere(local_skel_labels > 0)
            skel_ids = local_skel_labels[
                skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]
            ]
            tree = cKDTree(skel_coords)

            disp = vectors[:, coords[:, 0], coords[:, 1], coords[:, 2]].T
            embedded = coords + disp
            _, idx = tree.query(embedded)
            segmentation[coords[:, 0], coords[:, 1], coords[:, 2]] = (
                skel_ids[idx] + next_id - 1
            )
            next_id += num_local_skel

        unique_increment = chunk_num_voxels * pymorton.interleave(*chunk_corner)
        segmentation[segmentation > 0] += unique_increment
        segmentation = segmentation.astype(np.uint64)

        return np.expand_dims(segmentation, axis=0)

    @property
    def dtype(self):
        return np.uint64

    @property
    def is_segmentation(self):
        return True

    @property
    def num_channels(self):
        return 1


class SkeletonDistancePostprocessor(PostProcessor):
    """Turn an already-decoded 3-channel skeleton_distance head (semantic,
    skeleton, distance-from-skeleton -- channels 0-1 true [0,1] probabilities,
    channel 2 distance in voxels; see a `SkeletonDistanceDecodeWrapper` model
    wrapper analogous to `SkootsDecodeWrapper`) into an instance segmentation
    for a single block.

    Sibling of `SkootsPostprocessor` for the 3-channel scalar-distance head
    (see `skoots_loss.SkeletonDistanceLoss`'s docstring for why that head
    exists: no per-voxel direction, only distance, appropriate when
    instances don't actually touch). Same per-foreground-component gating
    and provisional-id/`SimpleBlockwiseMerger` chaining story as
    `SkootsPostprocessor` -- see that class's docstring for the full
    blockwise-stitching caveats, which apply unchanged here.

    Algorithm, per foreground connected-component in this block:
      1. threshold semantic/skeleton channels into binary masks.
      2. label the skeleton mask *restricted to that component* -- each
         skeleton connected-component is one candidate instance seed.
      3. if the component has no skeleton seed at all, keep it as one
         provisional instance (see `SkootsPostprocessor` point 3).
      4. otherwise, marker-controlled watershed: flood outward from each
         skeleton seed, walking uphill through the predicted
         distance-from-skeleton surface, restricted to this component. A
         voxel joins whichever seed's flood reaches it first -- i.e.
         whichever skeleton component the network estimates it's closest to
         the medial axis of. This is the classic distance-transform-watershed
         instance-splitting technique; it has no way to disambiguate two
         instances by *direction* the way `SkootsPostprocessor`'s vector
         embedding does, so it only works when this component's shape
         already implies which points belong to which skeleton (true here
         because these instances don't touch -- see corrections/skoots_mito
         investigation).
    """

    def __init__(
        self,
        semantic_channel: int = 0,
        skeleton_channel: int = 1,
        distance_channel: int = 2,
        semantic_threshold: float = 0.5,
        skeleton_threshold: float = 0.5,
    ):
        self.semantic_channel = int(semantic_channel)
        self.skeleton_channel = int(skeleton_channel)
        self.distance_channel = int(distance_channel)
        self.semantic_threshold = float(semantic_threshold)
        self.skeleton_threshold = float(skeleton_threshold)

    def _process(self, data, chunk_corner, chunk_num_voxels):
        data = data.astype(np.float32)
        semantic_prob = data[self.semantic_channel]
        skeleton_prob = data[self.skeleton_channel]
        distance = data[self.distance_channel]

        fg_mask = semantic_prob > self.semantic_threshold
        skel_mask = (skeleton_prob > self.skeleton_threshold) & fg_mask

        fg_components, num_fg = label(fg_mask)
        segmentation = np.zeros(fg_mask.shape, dtype=np.int64)
        next_id = 1

        for comp_id in range(1, num_fg + 1):
            comp_mask = fg_components == comp_id

            local_skel_labels, num_local_skel = label(comp_mask & skel_mask)
            if num_local_skel == 0:
                # true skeleton not visible in this block -- provisional id,
                # left for SimpleBlockwiseMerger (or a future offline
                # stitching pass) to reconcile with a neighboring block.
                segmentation[comp_mask] = next_id
                next_id += 1
                continue

            markers = np.where(comp_mask, local_skel_labels, 0)
            grown = watershed(distance, markers=markers, mask=comp_mask)
            segmentation[comp_mask] = grown[comp_mask] + next_id - 1
            next_id += num_local_skel

        unique_increment = chunk_num_voxels * pymorton.interleave(*chunk_corner)
        segmentation[segmentation > 0] += unique_increment
        segmentation = segmentation.astype(np.uint64)

        return np.expand_dims(segmentation, axis=0)

    @property
    def dtype(self):
        return np.uint64

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
        # Base to_dict() serializes every non-underscore instance attr, but
        # this class stores accumulated runtime state (equivalences, the
        # face dict, etc.) as plain attrs -- not to_dict()'s fault in
        # general, just this class's -- because `equivalences` specifically
        # needs to stay a plain public attribute for server.py's duck-typed
        # `hasattr(postprocess, "equivalences")` polling. Overriding here
        # instead: only the real constructor params round-trip through
        # to_dict() -> URL -> deserialize_list() -> __init__(**kwargs)
        # (see server.py's `refresh_dataset`, called on every request) --
        # everything else would fail with "unexpected keyword argument"
        # since __init__ doesn't accept it back.
        return {
            "name": self.name(),
            "channel": self.channel,
            "face_erosion_iterations": self.face_erosion_iterations,
        }

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

    # def to_dict(self):
    #     return {"name": self.name()}

    @property
    def num_channels(self):
        return len(self.channels)


class LambdaPostprocessor(PostProcessor):
    def __init__(self, expression: str):
        self.expression = expression
        self._lambda = eval(f"lambda x: {expression}")

    def _process(self, data) -> np.ndarray:
        return self._lambda(data.astype(np.float32))

    # def to_dict(self):
    #     return {"name": self.name(), "expression": self.expression}

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


PostProcessorMethods = [f for f in PostProcessor.__subclasses__()]
