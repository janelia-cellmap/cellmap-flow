"""SKOOTS-style training target generation from instance-labeled volumes.

Reproduces (not reinterprets) the target construction used by the official
SKOOTS implementation (github.com/buswinka/skoots, github.com/buswinka/bism),
verified against source rather than the paper's simplified prose:

- Skeleton: per-instance `skimage.morphology.skeletonize(method="lee")`,
  falling back to the centroid when an instance skeletonizes to nothing
  (`skoots/train/generate_skeletons.py::calculate_skeletons`).
- Skeleton mask target: a binary ball of fixed radius stamped around every
  skeleton voxel (`skoots/lib/utils.py::get_cached_disk_coords` +
  `skoots/lib/skeleton.py::skeleton_to_mask`). The official code stamps a 2D
  disk per z-plane with a smaller "flank" disk on adjacent planes, which is
  an anisotropy shortcut; here we stamp a genuine 3D ball since our data is
  much closer to isotropic and a ball is the isotropic limit of that trick.
- Vector field target: for every foreground voxel, the vector to the
  *nearest point on that voxel's own instance's skeleton*
  (`skoots/lib/skeleton.py::bake_skeleton`, via anisotropy-weighted nearest
  neighbor), then averaged with a 3x3x3 box filter
  (`average_baked_skeletons`) for a smooth transition near skeleton voxels.
  Background voxels get a zero vector target and are excluded from the
  vector loss by the semantic mask, matching the official masking behavior.

Validated against synthetic geometries (sphere, long rod, curved tube,
branched shape, two touching spheres) and real dense mitochondria
ground-truth crops (jrc_axolotl-heart-1 mito002/mito005) -- 100% boundary
vector accuracy at touching/close instance pairs -- before being wired into
the finetuning pipeline. See experiments/skoots/ for the validation scripts.
"""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

_NEIGHBOR_OFFSETS_26 = [
    o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)
]


def _prune_skeleton_points(coords: np.ndarray, min_branch_length: float) -> np.ndarray:
    """Remove short terminal spurs from a voxel skeleton's point cloud.

    `skimage.skeletonize` on a voxelized (staircase) surface routinely
    produces small 1-3 voxel side-spurs off the real medial path -- an
    artifact of the surface's stair-steps, not real branch topology. This
    builds a 26-connectivity graph over the skeleton voxels and iteratively
    removes the shortest terminal branch (the path from a degree-1 endpoint
    to the nearest branch point) while its length is under
    `min_branch_length`, same rule mesh-n-bone's `CustomSkeleton.prune` uses
    for mesh-derived skeletons: a branch only counts as a prunable spur when
    *exactly one* end is a branch point. A simple branch-point-free path
    (the object's entire skeleton, e.g. a short straight rod) is never
    touched even if short, since both its ends are endpoints, not spurs off
    something else.
    """
    if coords.shape[0] < 3:
        return coords
    coords_int = coords.astype(int)
    coord_to_idx = {tuple(c): i for i, c in enumerate(coords_int)}

    g = nx.Graph()
    g.add_nodes_from(range(len(coords_int)))
    for i, c in enumerate(coords_int):
        for o in _NEIGHBOR_OFFSETS_26:
            j = coord_to_idx.get((c[0] + o[0], c[1] + o[1], c[2] + o[2]))
            if j is not None:
                g.add_edge(i, j)

    while g.number_of_nodes() > 1:
        degrees = dict(g.degree())
        branchpoints = {n for n, d in degrees.items() if d > 2}
        endpoints = [n for n, d in degrees.items() if d <= 1]
        if not endpoints:
            break

        best = None  # (length, nodes_to_remove)
        for ep in endpoints:
            path = [ep]
            prev, cur = None, ep
            while True:
                nbrs = [n for n in g.neighbors(cur) if n != prev]
                if len(nbrs) != 1:
                    break
                prev, cur = cur, nbrs[0]
                path.append(cur)
                if degrees[cur] != 2:
                    break
            other_end = path[-1]
            if other_end == ep or other_end not in branchpoints:
                continue  # isolated point, or both ends are endpoints -- not a spur
            length = sum(
                np.linalg.norm(coords_int[path[k + 1]] - coords_int[path[k]])
                for k in range(len(path) - 1)
            )
            if length < min_branch_length and (best is None or length < best[0]):
                best = (length, path[:-1])  # keep the branch point, drop the spur

        if best is None:
            break
        _, to_remove = best
        if g.number_of_nodes() - len(to_remove) < 1:
            break
        g.remove_nodes_from(to_remove)

    return coords[sorted(g.nodes())]


def instance_skeletons(
    instance_labels: np.ndarray, pad: int = 2, min_branch_length: float = 3.0
) -> dict[int, np.ndarray]:
    """Per-instance skeleton voxel coordinates, keyed by instance id (>0).

    Falls back to the instance centroid when `skeletonize` returns nothing
    (e.g. an object only 1-2 voxels wide), matching the official fallback.

    Crops to each instance's own bounding box (+`pad`) before skeletonizing
    -- calling `skeletonize` on the full volume once per instance is fine
    for small toy volumes but pathological on a production-scale crop with
    dozens-to-hundreds of instances (e.g. a 600^3 volume with 101 objects
    would do 101 full-volume scans instead of 101 small local ones).

    `min_branch_length` (voxels) prunes short terminal spurs from the raw
    skeletonize output -- see `_prune_skeleton_points`. Set to 0 to disable.
    """
    skeletons = {}
    slices_by_id = ndi.find_objects(instance_labels)  # index i -> slices for label i+1
    for inst_id in np.unique(instance_labels):
        if inst_id == 0:
            continue
        box = slices_by_id[int(inst_id) - 1]
        if box is None:
            continue
        padded = tuple(
            slice(max(0, s.start - pad), min(dim, s.stop + pad))
            for s, dim in zip(box, instance_labels.shape)
        )
        local_mask = instance_labels[padded] == inst_id
        skel = skeletonize(local_mask, method="lee")
        coords = np.argwhere(skel > 0)
        if coords.shape[0] == 0:
            coords = np.argwhere(local_mask).mean(axis=0, keepdims=True)
        elif min_branch_length > 0:
            coords = _prune_skeleton_points(coords, min_branch_length)
        origin = np.array([s.start for s in padded])
        skeletons[int(inst_id)] = coords.astype(np.float64) + origin
    return skeletons


def skeleton_mask_target(
    instance_labels: np.ndarray,
    skeletons: dict[int, np.ndarray],
    radius: int = 2,
) -> np.ndarray:
    """Binary skeleton probability target: a ball of `radius` around each
    skeleton voxel, clipped to that voxel's *own instance's* mask, unioned
    across all instances.

    The official default (`SKELETON_MASK_RADIUS=9`) is tuned for that
    project's own voxel size/object spacing; at this dataset's 16nm voxel
    size a 9-voxel radius is an ~288nm-diameter ball, easily bridging two
    separate (but merely nearby, not touching) mitochondria's skeleton
    masks into one connected component -- which then makes postprocessing
    wrongly merge them into a single instance. `radius=2` (~64nm diameter)
    trades some skeleton-mask learnability for far fewer false bridges;
    revisit if this dataset's objects are packed even closer than that.

    Clipping per-instance (rather than one global ball union) matters even
    at radius=2: a skeleton point near an object's own boundary would
    otherwise stamp a ball that bleeds into background (or a neighboring
    instance) past that boundary -- the skeleton target should never claim
    "mitochondria" where the semantic target says there isn't one. Computed
    per-instance over each instance's own bounding box (+pad for the ball
    radius) rather than one whole-volume EDT, matching the other per-instance
    helpers in this module (`instance_skeletons`, `bake_vector_targets`).
    """
    mask = np.zeros(instance_labels.shape, dtype=bool)
    if not skeletons:
        return mask
    boxes = ndi.find_objects(instance_labels)
    for inst_id, coords in skeletons.items():
        box = boxes[inst_id - 1] if inst_id - 1 < len(boxes) else None
        if box is None:
            continue
        padded = tuple(
            slice(max(0, s.start - radius), min(dim, s.stop + radius))
            for s, dim in zip(box, instance_labels.shape)
        )
        origin = np.array([s.start for s in padded])
        local_shape = tuple(s.stop - s.start for s in padded)
        local_coords = (coords.round().astype(int) - origin)
        local_coords = local_coords[
            np.all((local_coords >= 0) & (local_coords < local_shape), axis=1)
        ]
        if local_coords.shape[0] == 0:
            continue
        points_mask = np.zeros(local_shape, dtype=bool)
        points_mask[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]] = True
        ball = ndi.distance_transform_edt(~points_mask) <= radius
        ball &= instance_labels[padded] == inst_id
        mask[padded] |= ball
    return mask


def _geodesic_nearest_indices(local_mask: np.ndarray, local_coords: np.ndarray) -> np.ndarray:
    """For every voxel in `local_mask`, the index into `local_coords` (a
    (K, 3) int array of seed-point voxel coordinates, all lying inside
    `local_mask`) of the seed nearest under a *geodesic* metric -- i.e. the
    seed reachable via the shortest path that stays entirely inside
    `local_mask` -- never a seed that's merely closest in a straight line
    through background.

    Implemented as multi-source BFS via `skimage.segmentation.watershed` on
    a flat (constant) image with one unique marker per seed and a hard
    `mask`: on a flat image, watershed's flood order is exactly a
    multi-source breadth-first search, so each masked voxel is claimed by
    whichever seed's flood front reaches it first -- and that front can
    never cross an unmasked (background) voxel. This is the standard
    "geodesic Voronoi via watershed" trick, and needs no new dependency
    (`skimage.segmentation.watershed` is already used by
    cellmap_flow/post/postprocessors.py).

    Returns an int array shaped like `local_mask`: each masked voxel holds
    the 0-based index of its geodesically nearest seed, and every other
    voxel (including any masked voxel geodesically unreachable from every
    seed -- e.g. a same-instance-labeled fragment disconnected, under this
    mask's connectivity, from all of its own skeleton points) holds -1,
    since watershed leaves unreached regions at label 0. Callers must
    handle -1 themselves.

    Pure graph/step-count BFS -- doesn't weight by anisotropy. Acceptable
    since every caller in this module runs with the default isotropic
    `anisotropy=(1, 1, 1)` in practice; revisit with
    `skimage.graph.MCP_Geometric` (which supports an anisotropic `sampling`)
    if a genuinely anisotropic dataset ever needs exact geodesic distances
    rather than just correct connectivity.
    """
    markers = np.zeros(local_mask.shape, dtype=np.int64)
    marker_ids = np.arange(1, len(local_coords) + 1)
    markers[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]] = marker_ids
    flat_image = np.zeros(local_mask.shape, dtype=np.uint8)
    labels = watershed(flat_image, markers=markers, mask=local_mask, connectivity=3)
    return labels.astype(np.int64) - 1


def bake_vector_targets(
    instance_labels: np.ndarray,
    skeletons: dict[int, np.ndarray],
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    smooth: bool = True,
    smoothing_agreement_voxels: float = 3.0,
) -> np.ndarray:
    """Vector field target: for each foreground voxel, the displacement to
    the nearest point on its own instance's skeleton, in voxel units.

    Shape (3, Z, Y, X), channel order (dz, dy, dx). `anisotropy` scales
    distance *ranking only* (matches `bake_skeleton`'s use of anisotropy to
    pick the nearest skeleton point under physical, not voxel, distance) --
    the returned vector itself stays in voxel units.

    "Nearest" is resolved *geodesically*, via `_geodesic_nearest_indices`:
    the chosen skeleton point is reachable from this voxel without ever
    leaving this voxel's own instance mask. A plain Euclidean nearest-point
    query (e.g. a raw `cKDTree` over the skeleton point cloud) can pick a
    point that's close in a straight line but on the *other side* of a gap
    -- e.g. a voxel near the narrow opening of a horseshoe/self-touching
    mitochondrion resolving to a skeleton point across the gap rather than
    the (farther, but actually-connected) point on its own side. That
    doesn't just look wrong locally: it also means two Euclidean-adjacent
    voxels on either side of the gap can resolve to skeleton points that
    are physically far apart, which is exactly the discontinuity the
    smoothing guard below has to defend against. Falls back to a plain
    (Euclidean) `cKDTree` query only for the rare voxel geodesically
    unreachable from every one of its own instance's skeleton points (see
    `_geodesic_nearest_indices`'s docstring) -- better than emitting a zero
    vector there.

    Smoothing (`smooth=True`) averages each voxel's *resolved skeleton
    point* with its 3x3x3-neighborhood's resolved points, then subtracts
    this voxel's own position at the end. A neighbor only contributes if it
    (a) belongs to the same instance and (b) resolved to a skeleton point
    within `smoothing_agreement_voxels` of this voxel's own resolved point.
    The official `average_baked_skeletons` has neither guard -- it convolves
    the whole multi-instance volume, so two nearby-but-different instances'
    skeleton positions bleed into each other right at their shared boundary.
    Guard (a) alone fixes that; note it also already rules out smoothing
    ever crossing empty space in the first place, since two Euclidean-
    adjacent voxels that are both foreground of the same instance are, by
    construction, one geodesic hop apart -- there's no "adjacent but not
    geodesically connected" case for a 3x3x3 (radius-1) neighborhood. Guard
    (b) is a secondary safety net for the remaining case geodesic
    resolution doesn't fully remove: right at a skeleton branch point, two
    adjacent voxels can still legitimately resolve to different branches.
    """
    shape = instance_labels.shape
    vectors = np.zeros((3,) + shape, dtype=np.float32)
    nearest_coord = np.zeros((3,) + shape, dtype=np.float32)
    aniso = np.asarray(anisotropy, dtype=np.float64)
    boxes = ndi.find_objects(instance_labels)  # one pass, not one full-volume scan per instance

    for inst_id, coords in skeletons.items():
        box = boxes[inst_id - 1] if inst_id - 1 < len(boxes) else None
        if box is None:
            continue
        origin = np.array([s.start for s in box])
        local_mask = instance_labels[box] == inst_id
        fg_local = np.argwhere(local_mask)
        if fg_local.shape[0] == 0:
            continue
        fg = fg_local + origin

        local_coords_f = coords - origin
        in_bounds = np.all(
            (local_coords_f >= 0) & (local_coords_f < np.array(local_mask.shape)), axis=1
        )
        local_coords_int = local_coords_f[in_bounds].round().astype(int)
        coords_in_bounds = coords[in_bounds]

        reached = np.zeros(fg.shape[0], dtype=bool)
        nearest = np.empty((fg.shape[0], 3), dtype=np.float64)
        if local_coords_int.shape[0] > 0:
            idx_field = _geodesic_nearest_indices(local_mask, local_coords_int)
            idx = idx_field[fg_local[:, 0], fg_local[:, 1], fg_local[:, 2]]
            reached = idx >= 0
            nearest[reached] = coords_in_bounds[idx[reached]]

        if not reached.all():
            # Geodesically unreachable from every one of this instance's
            # own skeleton points (or none of `coords` even fell inside
            # `box`) -- fall back to the old Euclidean nearest-point query
            # rather than leaving a zero vector.
            tree = cKDTree(coords * aniso)
            _, fallback_idx = tree.query(fg[~reached] * aniso)
            nearest[~reached] = coords[fallback_idx]

        vectors[:, fg[:, 0], fg[:, 1], fg[:, 2]] = (nearest - fg).T  # voxel-space displacement
        nearest_coord[:, fg[:, 0], fg[:, 1], fg[:, 2]] = nearest.T  # absolute resolved point

    if not smooth:
        return vectors

    fg_mask = instance_labels > 0
    z, y, x = shape
    padded_coord = np.pad(nearest_coord, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="constant")
    padded_labels = np.pad(instance_labels, 1, mode="constant")

    sum_coord = np.zeros((3,) + shape, dtype=np.float32)
    count = np.zeros(shape, dtype=np.float32)
    for dz, dy, dx in itertools.product((-1, 0, 1), repeat=3):
        sl = (slice(1 + dz, 1 + dz + z), slice(1 + dy, 1 + dy + y), slice(1 + dx, 1 + dx + x))
        neighbor_coord = padded_coord[(slice(None),) + sl]
        neighbor_labels = padded_labels[sl]

        same_instance = (neighbor_labels == instance_labels) & fg_mask
        agrees = np.linalg.norm(neighbor_coord - nearest_coord, axis=0) <= smoothing_agreement_voxels
        compatible = same_instance & agrees

        sum_coord += np.where(compatible[None], neighbor_coord, 0.0)
        count += compatible

    count[count == 0] = 1.0
    smoothed_coord = sum_coord / count

    zz, yy, xx = np.meshgrid(
        np.arange(z, dtype=np.float32),
        np.arange(y, dtype=np.float32),
        np.arange(x, dtype=np.float32),
        indexing="ij",
    )
    position = np.stack([zz, yy, xx], axis=0)
    return (smoothed_coord - position) * fg_mask[None]


def skeleton_distance_target(
    instance_labels: np.ndarray,
    skeletons: dict[int, np.ndarray],
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Scalar distance-to-nearest-own-skeleton-point target (voxel units).

    Delegates to `bake_vector_targets` (unsmoothed) and keeps only the
    displacement magnitude rather than the vector -- a 1-channel regression
    target instead of 3, sharing that function's geodesic (mask-respecting)
    nearest-skeleton-point resolution rather than duplicating a second,
    separately-maintainable copy of it. Dropping direction loses the vector
    field's ability to *disambiguate* two nearby instances' skeletons (which
    is what actually lets SKOOTS's embedding trick split touching objects);
    it's the right tradeoff only when instances don't truly touch in the
    first place, so a plain distance-transform-style watershed seeded from
    skeleton components is enough to recover per-instance shape. Background
    stays 0, matching `bake_vector_targets`.
    """
    vectors = bake_vector_targets(instance_labels, skeletons, anisotropy=anisotropy, smooth=False)
    return np.linalg.norm(vectors, axis=0).astype(np.float32)


def build_skeleton_distance_targets(
    instance_labels: np.ndarray,
    skeleton_radius: int = 2,
    min_branch_length: float = 3.0,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, np.ndarray]:
    """Convenience wrapper: instance labels -> [semantic, skeleton, distance]."""
    skeletons = instance_skeletons(instance_labels, min_branch_length=min_branch_length)
    semantic = (instance_labels > 0).astype(np.float32)
    skeleton = skeleton_mask_target(instance_labels, skeletons, radius=skeleton_radius).astype(np.float32)
    distance = skeleton_distance_target(instance_labels, skeletons, anisotropy=anisotropy)
    return {
        "semantic": semantic,
        "skeleton": skeleton,
        "distance": distance,
        "skeleton_points": skeletons,
    }


def stack_skeleton_distance_target_and_mask(
    instance_labels: np.ndarray, targets: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Pack [semantic, skeleton, distance] into the (3, Z, Y, X) layout
    `SkeletonDistanceLoss` / `SkeletonDistanceTargetTransform` expect.

    See `stack_targets_and_mask`'s docstring re: the all-ones dense-crop mask
    assumption -- identical here, just 3 channels instead of 5.
    """
    target = np.stack(
        [targets["semantic"], targets["skeleton"], targets["distance"]], axis=0
    ).astype(np.float32)
    annotated = np.ones_like(targets["semantic"], dtype=np.float32)
    mask = np.repeat(annotated[None], 3, axis=0).astype(np.float32)
    return target, mask


def build_skeleton_semantic_targets(
    instance_labels: np.ndarray,
    skeleton_radius: int = 2,
    min_branch_length: float = 3.0,
) -> dict[str, np.ndarray]:
    """Convenience wrapper: instance labels -> [semantic, skeleton] only.

    Skips the vector/distance computation entirely (unlike
    `build_skoots_targets`/`build_skeleton_distance_targets`, which compute
    it and then have it discarded downstream) -- for a head with no
    instance-splitting channel at all, there's nothing to bake.
    """
    skeletons = instance_skeletons(instance_labels, min_branch_length=min_branch_length)
    semantic = (instance_labels > 0).astype(np.float32)
    skeleton = skeleton_mask_target(instance_labels, skeletons, radius=skeleton_radius).astype(np.float32)
    return {"semantic": semantic, "skeleton": skeleton, "skeleton_points": skeletons}


def stack_skeleton_semantic_target_and_mask(
    instance_labels: np.ndarray, targets: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Pack [semantic, skeleton] into the (2, Z, Y, X) layout
    `SkeletonSemanticLoss` / `SkeletonSemanticTargetTransform` expect.

    See `stack_targets_and_mask`'s docstring re: the all-ones dense-crop mask
    assumption -- identical here, just 2 channels.
    """
    target = np.stack([targets["semantic"], targets["skeleton"]], axis=0).astype(np.float32)
    annotated = np.ones_like(targets["semantic"], dtype=np.float32)
    mask = np.repeat(annotated[None], 2, axis=0).astype(np.float32)
    return target, mask


def build_skoots_targets(
    instance_labels: np.ndarray,
    skeleton_radius: int = 2,
    min_branch_length: float = 3.0,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, np.ndarray]:
    """Convenience wrapper: instance labels -> the three SKOOTS targets."""
    skeletons = instance_skeletons(instance_labels, min_branch_length=min_branch_length)
    semantic = (instance_labels > 0).astype(np.float32)
    skeleton = skeleton_mask_target(instance_labels, skeletons, radius=skeleton_radius).astype(np.float32)
    vectors = bake_vector_targets(instance_labels, skeletons, anisotropy=anisotropy)
    return {
        "semantic": semantic,
        "skeleton": skeleton,
        "vectors": vectors,
        "skeleton_points": skeletons,
    }


def stack_targets_and_mask(
    instance_labels: np.ndarray, targets: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Pack the three target arrays into the (5, Z, Y, X) layout `SkootsLoss`
    and `SkootsTargetTransform` expect: [semantic, skeleton, vec_z, vec_y, vec_x].

    Mask channels 0-1 (semantic, skeleton) and 2-4 (vector, all three sharing
    one mask) are both "annotated foreground/background" -- everything here
    assumes `instance_labels` comes from a *dense* annotation (every voxel is
    either background or a real instance), so the mask is simply "was this
    crop annotated at all", i.e. all-ones for a fully dense crop. Sparse
    crops (unannotated=0) should pass their own annotated mask in via
    `annotated`.
    """
    target = np.concatenate(
        [targets["semantic"][None], targets["skeleton"][None], targets["vectors"]], axis=0
    ).astype(np.float32)
    annotated = np.ones_like(targets["semantic"], dtype=np.float32)
    mask = np.concatenate(
        [annotated[None], annotated[None], np.repeat(annotated[None], 3, axis=0)], axis=0
    ).astype(np.float32)
    return target, mask
