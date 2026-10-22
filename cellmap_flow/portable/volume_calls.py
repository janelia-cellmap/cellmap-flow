"""What the browser calls to plan an annotation volume and its good regions:
the functions ``web/src/finetune/volume.ts``, ``web/src/finetune/good-regions.ts``
and ``web/src/finetune/run.ts`` run through chunkmirage's
``Engine.call("cellmap_flow.portable.volume_calls:<name>", args, arrays)``
(``calls.py`` says the shape). Each takes plain numbers in its JSON and
answers JSON; the rules are ``volume.py``'s and ``regions.py``'s, which the
server calls in its process. A refusal (a ValueError of those rules) is
answered as ``{"error": <its sentence>}``, so the page shows the sentence the
server's route would, not a Python traceback.
"""

import functools

from cellmap_flow.portable import regions, volume


def _refusal(fn):
    """``fn`` answering ``{"error": <sentence>}`` where it raises a ValueError."""

    @functools.wraps(fn)
    def call(args, arrays):
        try:
            return fn(args, arrays)
        except ValueError as e:
            return {"error": str(e)}

    return call


@_refusal
def plan_volume(args, arrays):
    """``volume.plan_volume`` of ``level`` (``shape``, ``voxel``, ``corner``:
    the raw level the model reads) and ``model`` (its voxel sizes and its
    blocks, as ``block_shapes`` takes them), with ``dtype``, the labels'
    (``volume.annotation_dtype`` of ``output_type``, what the model trains on)."""
    plan = volume.plan_volume(args["level"], args["model"])
    return {**plan, "dtype": volume.annotation_dtype(args.get("output_type"))}


def _layer_boxes(entries) -> list:
    """Each entry as the good_regions layer draws it: its box and label
    (``good`` without a label), or None for one that is not a region."""
    out = []
    for entry in entries:
        box = regions.region_box_nm(entry)
        if box is None:
            out.append(None)
            continue
        label = entry.get("label")
        out.append({"offset_nm": box[0].tolist(), "shape_nm": box[1].tolist(),
                    "label": label if isinstance(label, str) else "good"})
    return out


@_refusal
def mark_region(args, arrays):
    """The region a mark makes (``regions.new_region``): a box of the size
    ``size_nm`` asks for (``regions.region_size_nm``, the volume's output
    block by default: ``volume`` has its ``chunk_size`` and
    ``output_voxel_size``, or is null) centred on ``centre_nm``, labelled
    ``label`` or after the ``regions`` already stored. Answers the region,
    and ``boxes``, the layer's boxes of the stored regions and it
    (``region_boxes``)."""
    grid = args.get("volume") or {}
    size = regions.region_size_nm(args.get("size_nm"), grid.get("chunk_size"), grid.get("output_voxel_size"))
    stored = list(args.get("regions") or [])
    region = regions.new_region(args["centre_nm"], size, len(stored), args.get("label"))
    return {"region": region, "boxes": _layer_boxes([*stored, region])}


def region_boxes(args, arrays):
    """The good_regions layer's boxes of ``regions``: per entry its
    ``offset_nm``, ``shape_nm`` and ``label``, or null for one that is not a
    region (``regions.region_box_nm``), so the layer's ids stay the entries'
    indices."""
    return _layer_boxes(args.get("regions") or [])


def anchor_boxes(args, arrays):
    """``regions.anchor_boxes`` of ``regions`` on the grid of ``volume``
    (its ``dataset_offset_nm``, ``output_voxel_size``, ``dataset_shape_voxels``
    and ``chunk_size``, the model's output block): the boxes, and the
    entries skipped, each by its index in ``regions``."""
    grid = args["volume"]
    corner = volume.volume_corner_nm(grid.get("dataset_offset_nm"), grid["output_voxel_size"])
    return regions.anchor_boxes(
        args.get("regions") or [], corner, grid["output_voxel_size"], grid["dataset_shape_voxels"], grid["chunk_size"]
    )
