"""``BoundingBox``: a box of a dataset to process, as a blockwise task lists it.

A task YAML's ``bounding_boxes`` is a list of them, which the blockwise
processor checks (``_check_bounding_boxes``); the dashboard's box tool draws
and loads the INPUT node's (``dashboard.requests.BbxGenerator``), which the
task YAML it generates lists. Importing this imports pydantic only: no Flask,
and none of the blockwise processor.
"""

from typing import Annotated

from pydantic import BaseModel, Field

# z, y, x, nm.
_Point = Annotated[list[float], Field(min_length=3, max_length=3)]
_Size = Annotated[list[Annotated[float, Field(gt=0)]], Field(min_length=3, max_length=3)]


class BoundingBox(BaseModel):
    """A box, nm, z, y, x: ``offset`` its lower corner (0, 0, 0 when left
    out) and ``shape`` its size, every side above 0."""

    offset: _Point = [0, 0, 0]
    shape: _Size
