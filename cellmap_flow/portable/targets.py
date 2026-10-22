"""The paint as training targets: what a finetune trains a model toward.

Labels mean what the paint layer paints: 0 unpainted (no loss), 1
background, 2 and up objects (an instance volume numbers each one). Each
function takes the labels of a patch, (Z, Y, X), or of a batch of them with
any leading axes, (..., Z, Y, X), and answers float32 arrays with a channel
axis in front of the three spatial ones, (..., C, Z, Y, X): one channel per
output channel of the model, the target and the mask of the voxels its loss
counts. The server's ``finetune/target_transforms.py`` wraps these for the
trainer's (B, 1, Z, Y, X) tensors; the browser calls them on one painted
chunk through ``targets_calls.py``.

What a model trains toward is the caller's to say: the binary target for a
model of objects, the affinities along ``offsets`` for an affinity model
(the server reads them from the model's script or its channel names, the
browser from the names in its manifest), and for a distance model the soft
signed distance (dense paint) or the bounds on it (scribbles), on the
server only for now.
"""

import numpy as np

# How far out, in sigma, a distance bound still says anything: tanh(3) is
# 0.995, so a voxel known to be at least 3 sigma inside is as good as at 3
# sigma, and an upper bound beyond it constrains nothing a threshold sees.
SATURATION_SIGMAS = 3.0


def binary(labels, channels: int = 1):
    """``(target, mask)``: 1 where the voxel is an object (label 2 or more),
    and where it is painted (label above 0), the same on every one of
    ``channels`` channels.

    One channel is the target of a model of one class. More treat a
    multi-channel model (affinities, distances) as simple binary
    segmentation, every channel trained toward the same objects.
    """
    labels = np.asarray(labels)
    _check_patch(labels)
    _check_channels(channels)
    shape = (*labels.shape[:-3], channels, *labels.shape[-3:])
    target = np.broadcast_to((labels > 1)[..., None, :, :, :], shape).astype(np.float32)
    mask = np.broadcast_to((labels > 0)[..., None, :, :, :], shape).astype(np.float32)
    return target, mask


def affinity(labels, offsets, channels: int = None):
    """``(target, mask)``: channel c the affinity along ``offsets[c]``.

    A voxel's affinity is 1 when it and its neighbour at the offset are the
    same object (equal labels above 1), and 0 when they are different
    objects or either is background. It is written at the voxel, and masked
    only where both of the pair are painted, so each channel has a mask of
    its own.

    ``offsets`` are [dz, dy, dx], one for each of the first channels; an
    offset of None leaves its channel out of the loss, as are the channels
    past the offsets when ``channels`` (the model's output channels, by
    default one per offset) has more: LSDs, say, which then get no
    gradient. An offset as long as the patch along its axis or longer pairs
    no voxel inside it, so its channel has an empty mask: a one-slice
    patch's z channel, the head of a 2D student, keeps the model's weights.
    """
    labels = np.asarray(labels)
    _check_patch(labels)
    channels = len(offsets) if channels is None else channels
    _check_channels(channels)
    if len(offsets) > channels:
        raise ValueError(f"{len(offsets)} affinity offsets for {channels} channel(s): a model has a channel per offset")
    spatial = labels.shape[-3:]
    target = np.zeros((*labels.shape[:-3], channels, *spatial), np.float32)
    mask = np.zeros_like(target)
    for c, offset in enumerate(offsets):
        if offset is None:
            continue
        pairs = _pairs(spatial, offset)
        if pairs is None:
            continue
        here, there = pairs
        a, b = labels[(..., *here)], labels[(..., *there)]
        target[(..., c, *here)] = (a == b) & (a > 1)
        mask[(..., c, *here)] = (a > 0) & (b > 0)
    return target, mask


def _pairs(shape, offset):
    """The voxels whose neighbour at ``offset`` lies in a patch of ``shape``,
    and those neighbours, as two tuples of slices; None when there are none.

    Along each axis the voxels are [max(0, -d), min(n, n - d)) and their
    neighbours d further on. An offset of n or more along an axis of n
    voxels pairs nothing, which is said here rather than left to slicing,
    where a stop of n - d below 0 counts from the far end.
    """
    if len(offset) != 3 or any(int(d) != d for d in offset):
        raise ValueError(f"an affinity offset is three whole numbers (dz, dy, dx), not {offset!r}")
    if any(abs(int(d)) >= n for d, n in zip(offset, shape)):
        return None
    here = tuple(slice(max(0, -int(d)), min(n, n - int(d))) for d, n in zip(offset, shape))
    there = tuple(slice(max(0, int(d)), min(n, n + int(d))) for d, n in zip(offset, shape))
    return here, there


def distance(labels, sigma_voxels: float = 6.0, channels: int = 1):
    """``(target, mask)``: the soft signed distance of models trained the fly_organelles way.

    The target is ``(tanh((edt(fg) - edt(not fg)) / sigma) + 1) / 2`` with
    distances in voxels: 0.5 on the object boundary, rising to 1 inside and
    falling to 0 outside, saturating a few sigma away. That is exactly what
    ``fly_organelles.utils.Distance`` fed the cellmap distance models (the
    nuc/mito "*_distance_*" HuggingFace repos use sigma=6), so sigmoid(model
    output) is comparable to this target voxel for voxel. It goes with a
    BCE-with-logits loss, which accepts soft targets; margin and dice do not.
    The target is the same on every one of ``channels`` channels.

    Masking: an EDT computed inside a patch only sees boundaries inside the
    patch. For an annotated voxel the computed |d| is an upper bound on the
    truth -- the real nearest boundary may sit just past the patch edge or
    inside an unannotated pocket. A voxel is supervised only when its
    computed |d| is no larger than its distance to the nearest unannotated
    voxel or patch edge (so no unseen boundary can be closer), or when that
    distance is already past 3*sigma, where tanh has saturated and the exact
    value no longer matters.
    """
    labels = np.asarray(labels)
    _check_patch(labels)
    _check_channels(channels)
    sigma = _positive_sigma(sigma_voxels)
    lead, spatial = labels.shape[:-3], labels.shape[-3:]
    target = np.empty((*lead, 1, *spatial), np.float32)
    mask = np.empty_like(target)
    for at in np.ndindex(*lead):
        target[(*at, 0)], mask[(*at, 0)] = _distance_one(labels[at], sigma)
    shape = (*lead, channels, *spatial)
    return np.broadcast_to(target, shape).copy(), np.broadcast_to(mask, shape).copy()


def _distance_one(ann, sigma):
    """(Z, Y, X) labels -> (target, mask) float32, for ``distance``."""
    from scipy.ndimage import distance_transform_edt as edt

    annotated = ann > 0
    fg = ann >= 2
    bg = ann == 1
    target = np.zeros(ann.shape, dtype=np.float32)
    mask = np.zeros(ann.shape, dtype=np.float32)
    if not annotated.any():
        return target, mask

    # edt(x) is the distance from each nonzero voxel of x to the nearest
    # zero. With no zero anywhere scipy returns a large finite number for
    # every voxel; treat that as "no boundary in this patch" explicitly.
    #
    # Both sides measure to the nearest *known* voxel of the other class:
    # a background voxel's distance to annotated foreground, and a
    # foreground voxel's distance to annotated background. d_in used to
    # be edt(fg), which counts unannotated voxels as background too, so a
    # foreground voxel next to an unannotated one -- at the edge of a
    # dense crop that cuts through an object, say -- got a small d_in that
    # always passed the trust test below, and was supervised toward 0.58
    # to 0.88 as though the object ended at the crop edge.
    d_in = edt(~bg) if bg.any() else np.full(ann.shape, np.inf)
    d_out = edt(~fg) if fg.any() else np.full(ann.shape, np.inf)
    signed = np.where(fg, d_in, -d_out)

    # Distance from each annotated voxel to the nearest unannotated voxel
    # or to just outside the patch (the one-voxel pad of False).
    known = np.pad(annotated, 1, constant_values=False)
    trust = edt(known)[1:-1, 1:-1, 1:-1]

    reliable = np.abs(signed) <= trust
    saturated = trust >= 3.0 * sigma
    mask[annotated & (reliable | saturated)] = 1.0
    target[:] = (np.tanh(signed / sigma) + 1.0) / 2.0
    return target, mask


def interval(labels, sigma_voxels: float = 6.0, voxel_size_nm=None):
    """``(bounds, mask)``: bounds on a distance model's output at each painted voxel, for scribbles.

    A distance target needs every voxel's distance to the object boundary,
    and a stroke does not say where the boundary is. It does bound it
    (``distance_bounds``). The bounds are returned in the model's logit
    space, where the distance target (tanh(d/sigma) + 1)/2 = sigmoid(2d/sigma)
    is the line z = 2d/sigma, with the sign of the painted class. Past
    SATURATION_SIGMAS a lower bound is capped there and an upper bound
    dropped (+-inf).

    Distances are Euclidean in nm (``voxel_size_nm``, so an anisotropic
    grid is measured as it is; 1 nm a voxel when None), and sigma, which is
    in output voxels as for the distance target, is taken in the finest
    axis's voxels (``sigma_nm``).

    bounds has two channels, the lower and the upper bound on the logit (0
    off the paint), and mask one, the painted voxels, whatever the model's
    channels: losses.IntervalLoss broadcasts them over its output.
    """
    labels = np.asarray(labels)
    _check_patch(labels)
    voxel_size = _voxel_size(voxel_size_nm)
    sigma = sigma_nm(sigma_voxels, voxel_size)
    lead, spatial = labels.shape[:-3], labels.shape[-3:]
    bounds = np.empty((*lead, 2, *spatial), np.float32)
    mask = np.empty((*lead, 1, *spatial), np.float32)
    for at in np.ndindex(*lead):
        ann = labels[at]
        lower, upper = distance_bounds(ann, voxel_size)
        logit_per_nm = 2.0 / sigma
        saturation = SATURATION_SIGMAS * sigma
        near = logit_per_nm * np.minimum(lower, saturation)
        far = np.where(upper < saturation, logit_per_nm * upper, np.inf)
        fg, bg = ann >= 2, ann == 1
        # Background lies on the negative side: its bounds swap and change sign.
        bounds[(*at, 0)] = np.where(fg, near, np.where(bg, -far, 0.0))
        bounds[(*at, 1)] = np.where(fg, far, np.where(bg, -near, 0.0))
        mask[(*at, 0)] = ann > 0
    return bounds, mask


def sigma_nm(sigma_voxels: float, voxel_size_nm=None) -> float:
    """The interval target's sigma in nm: ``sigma_voxels`` of the finest axis's voxels."""
    return _positive_sigma(sigma_voxels) * min(_voxel_size(voxel_size_nm))


def distance_bounds(ann, voxel_size_nm=None):
    """(L, U) in nm for each voxel of a (Z, Y, X) annotation; 0 off the paint, U inf with no other class.

    - upper, U: the nearest voxel painted as the other class. The boundary
      lies between the two, so |d| <= U.
    - lower, L: the nearest voxel not painted as the voxel's own class --
      the other class, unpainted (it may be either), or past the patch edge.
      Every voxel closer is known to be its own class, so |d| >= L.

    Where the paint is dense, the nearest voxel not of its own class is of
    the other, and L == U: the exact distance ``distance`` would give, with
    the same convention (edt to the nearest voxel of the other class, so 1
    voxel on either side of a boundary). On a stroke the two bracket the
    truth. Both are computed in the patch the trainer gets, which keeps them
    valid: a closer other-class voxel outside the patch would only lower the
    true |d|, which U already exceeds or meets, and the patch edge counts as
    "not its own class", which can only lower L. Reading context around the
    patch would tighten bounds near its edges, but the dataset hands the
    trainer the patch alone (augmented with it), so they are left as loose
    as the patch makes them.
    """
    from scipy.ndimage import distance_transform_edt as edt

    ann = np.asarray(ann)
    sampling = _voxel_size(voxel_size_nm)
    fg, bg = ann >= 2, ann == 1
    lower = np.zeros(ann.shape)
    upper = np.zeros(ann.shape)
    for own, other in ((fg, bg), (bg, fg)):
        if not own.any():
            continue
        # edt(x) is the distance from each nonzero voxel of x to the
        # nearest zero; with no zero at all scipy returns garbage, so a
        # patch without the other class has no upper bound, explicitly.
        to_other = edt(~other, sampling=sampling) if other.any() else np.inf
        # Padded with a ring of "not own class": the patch's surroundings.
        to_not_own = edt(np.pad(own, 1, constant_values=False), sampling=sampling)[1:-1, 1:-1, 1:-1]
        upper[own] = np.broadcast_to(to_other, ann.shape)[own]
        lower[own] = to_not_own[own]
    return lower, upper


def _check_patch(labels):
    if labels.ndim < 3:
        raise ValueError(f"labels are a patch (z, y, x), or a batch of them, not an array of shape {labels.shape}")


def _check_channels(channels):
    if int(channels) != channels or channels < 1:
        raise ValueError(f"a target needs at least one channel, not {channels!r}")


def _positive_sigma(sigma_voxels) -> float:
    if sigma_voxels <= 0:
        raise ValueError(f"sigma_voxels must be positive, got {sigma_voxels}")
    return float(sigma_voxels)


def _voxel_size(voxel_size_nm) -> tuple:
    return tuple(float(v) for v in ((1.0, 1.0, 1.0) if voxel_size_nm is None else voxel_size_nm))
