"""A box of a dataset, nm, z, y, x: the box tool's (lenient) and a blockwise task's (strict). No Flask."""

from typing import Annotated

from pydantic import BaseModel, Field

_Triple = Annotated[list[float], Field(min_length=3, max_length=3)]


class BoundingBox(BaseModel):
    """Its lower corner and its size."""

    offset: _Triple = [0, 0, 0]
    shape: _Triple = [1, 1, 1]


class TaskBoundingBox(BoundingBox):
    """A task YAML's box: its shape given, every side above 0."""

    shape: Annotated[list[Annotated[float, Field(gt=0)]], Field(min_length=3, max_length=3)]
