"""A dataset path as a person pastes it, as cellmap-flow reads it.

A path copied from Neuroglancer's source field names its format as well as
its location: in front, as ``zarr://``, ``zarr2://``, ``zarr3://`` or
``n5://`` (``zarr://https://host/x.zarr/raw``), or after it in Neuroglancer's
newer pipeline form, as ``|zarr:``, ``|zarr2:``, ``|zarr3:``, ``|n5:`` or
``|neuroglancer-precomputed:`` (``gs://bucket/x.zarr/raw|zarr2:``), where
what follows the colon is a path inside the location
(``gs://bucket/x.zarr|zarr2:raw`` is ``gs://bucket/x.zarr/raw``).
cellmap-flow finds a zarr's or an N5's format itself, so ``pasted_path``
drops the format and keeps the location; a precomputed volume is named
``precomputed://<location>``, as cellmap-flow names it (``io/paths.py``,
the browser's ``web/src/backend/routes/data.ts``), and that form is kept as
it is. Blanks and quotes around the text go too. Anything else, a local
path or a URL (``gs://``, ``s3://``, ``https://``, ``file://``) or a format
cellmap-flow does not read (``graphene://``), is left as it is.

Called where a person gives a dataset: the CLIs' ``-d`` (``view``,
``infer <type>``, ``add --run``), a YAML's ``data_path`` (``config/yaml.py``
``load_config``, for ``cellmap_flow yaml`` and blockwise tasks), and the
dashboard's dataset field and box tool (``dashboard/requests.py`` SetData
and BbxGenerator). The browser's twin, ``web/src/shared/pasted.ts``
``pastedPath``, applies the same pattern, which ``build/export_data.py``
exports, to its dataset field (``routes/data.ts``) and its ``?dataset=``
links (``web/src/link.ts`` parseLink); both are held to one table of cases,
``web/test/fixtures/pasted_paths.json``.
"""

import re

# The text, split into the location, the format named after it (or None)
# and the path inside the location that follows that format's colon (or
# None). Written in the syntax JavaScript's RegExp shares, since the browser
# reads it (build/export_data.py).
NEUROGLANCER_SOURCE = re.compile(
    r"""^\s*["']?\s*(?:(?:zarr[23]?|n5)://)?(.*?)"""
    r"""(?:\|(zarr[23]?|n5|neuroglancer-precomputed):([^|#?]*?))?\s*["']?\s*$""",
    re.DOTALL,
)


def pasted_path(text: str) -> str:
    """The dataset ``text`` names, without the Neuroglancer format around it."""
    location, format_after, inner = NEUROGLANCER_SOURCE.match(text).groups()
    if inner:
        location = f"{location.rstrip('/')}/{inner}"
    return f"precomputed://{location}" if format_after == "neuroglancer-precomputed" else location
