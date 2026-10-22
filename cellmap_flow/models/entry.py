"""A model entry, and how its values become constructor arguments.

A model entry is ``{"type": <type name>, "name": ..., "env": ...,
**arguments}``: the one shape every description of a model takes. A YAML's
``models:`` holds them; ``cellmap_flow infer <type>`` makes one of its
options; the dashboard's model form posts one; the Models tab's catalog,
Hugging Face, zoo and Cellpose pickers build one; ``models.resolve`` returns
one for a pasted reference; ``ModelConfig.to_dict()`` writes one back; and
``cellmap_flow serve --model`` takes one. ``registry.build_model`` builds the
config from it, and ``coerce`` here is how the entry's values become the
constructor's arguments.

The values arrive as whatever their source has: numbers and lists from a
YAML or JSON, strings from the command line and the form. ``coerce`` reads
each constructor argument's annotation and makes the value fit it, so the
same entry gives the same config wherever it came from:

- ``int``, ``float``, ``bool``, ``str``: that type. A string is parsed ("8"
  is 8, "5.24" is 5.24, "true" is True); a whole float is an int.
- ``tuple`` or ``list``, with or without an element type (``tuple[int, ...]``,
  ``list[str]``): a sequence. A string is a JSON list or comma-separated
  values ("8,8,8", "(16, 8, 8)", "mito, er"), and one number is a one-item
  sequence. Each item takes the element type when the annotation gives one;
  without one, an item that reads as a number is a number and the rest stay
  strings.
- ``dict``: a JSON object from a string. A string that is not JSON is left as
  it is, for the constructor (a finetune's base64 ``base_model``, the form
  launchers used before 0.3.0).
- no annotation: the value as given.

None, a blank string (an empty form field) and an empty list are left out,
so the constructor's default applies, and names the constructor does not take are
dropped: an entry ``to_dict()`` wrote may carry more than the constructor
takes (a finetuned model's channels, for display).

The helpers after ``coerce`` are what the constructors normalize their
arguments with, whichever way they were called: through ``build_model``, or
from Python with a number, a tuple or a Coordinate. They parse text through
the same ``split``, so a string reaches the same value by either route.
"""

import inspect
import json
import re
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

# Nothing but the standard library: models.resolve imports this, and must
# not pull numpy in to say what a reference is.

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_NAMED_TYPES = {"int": int, "float": float, "bool": bool, "str": str, "tuple": tuple, "list": list, "dict": dict}


def number(value):
    """``value`` as an int when it is whole, else as a float.

    ``int()`` truncated a 5.24 nm voxel size to 5, which put crops and
    layers on the wrong grid.
    """
    value = float(value)
    return int(value) if value.is_integer() else value


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def split(text: str) -> list:
    """The items of a sequence written as text.

    A JSON list is read as JSON, so its items keep their types. Otherwise
    the text, less brackets or parentheses around it, is split on commas
    ("8,8,8", "(16, 8, 8)", "mito, er"), or on commas and spaces when every
    piece is a number ("8 8 8", as a voxel size is typed), into strings.
    """
    text = text.strip()
    if text.startswith("["):
        try:
            items = json.loads(text)
        except ValueError:
            items = None
        if isinstance(items, list):
            return items
    inner = text.strip("()[]").strip()
    if not inner:
        return []
    pieces = re.split(r"[\s,]+", inner)
    if all(_is_number(piece) for piece in pieces):
        return pieces
    return [piece.strip() for piece in inner.split(",") if piece.strip()]


def _is_sequence(value) -> bool:
    """Whether ``value`` is a sequence of items (a list, a tuple, a
    Coordinate, an array), not a scalar or a string."""
    if isinstance(value, (str, bytes, dict)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _item(value, kind):
    """One value as ``kind``: int, float, bool or str; None for a number
    when it reads as one, else the string.

    Raises:
        ValueError: a string that is not what ``kind`` asks for.
    """
    if not isinstance(value, str):
        if kind is int:
            whole = number(value)
            if not isinstance(whole, int):
                raise ValueError(f"must be a whole number, not {value!r}")
            return whole
        if kind is float:
            return float(value)
        if kind is bool:
            return bool(value)
        return value
    text = value.strip()
    if kind is bool:
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False
        raise ValueError(f"must be true or false, not {value!r}")
    if kind is str:
        return value
    if not _is_number(text):
        if kind is None:
            return value
        raise ValueError(f"must be a {'whole ' if kind is int else ''}number, not {value!r}")
    parsed = number(text)
    if kind is int and not isinstance(parsed, int):
        raise ValueError(f"must be a whole number, not {value!r}")
    return float(parsed) if kind is float else parsed


def _parse(annotation):
    """``(scalar kind, container, element kind)`` of a constructor argument's annotation.

    The scalar kind is int, float, bool or str, or None; the container is
    list, tuple or dict, or None; the element kind is a list or tuple's item
    type when the annotation names one. ``Optional[X]`` is read as ``X``.
    An annotation this cannot read leaves the value as given.
    """
    nothing = (None, None, None)
    if annotation is inspect.Parameter.empty or annotation is Any:
        return nothing
    if isinstance(annotation, str):
        # An annotation left as text: its name says enough.
        annotation = _NAMED_TYPES.get(annotation.split("[")[0].strip())
        if annotation is None:
            return nothing
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _parse(args[0]) if len(args) == 1 else nothing
    container = origin if origin in (list, tuple, dict) else annotation if annotation in (list, tuple, dict) else None
    if container in (list, tuple):
        args = [a for a in get_args(annotation) if a is not Ellipsis]
        element = args[0] if args and args[0] in (int, float, bool, str) else None
        return None, container, element
    if container is dict:
        return None, dict, None
    if annotation in (int, float, bool, str):
        return annotation, None, None
    return nothing


def coerce_value(value, annotation):
    """``value`` as a constructor argument annotated ``annotation`` (see the module docstring).

    Raises:
        ValueError: ``value`` cannot be read as what the annotation asks for.
    """
    kind, container, element = _parse(annotation)
    if container in (list, tuple):
        items = split(value) if isinstance(value, str) else list(value) if _is_sequence(value) else [value]
        items = [_item(item, element) for item in items]
        return tuple(items) if container is tuple else items
    if container is dict:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                return value
            return parsed if isinstance(parsed, dict) else value
        return value
    if kind is not None:
        return _item(value, kind)
    return value


def blank(value) -> bool:
    """Whether ``value`` says nothing: None, a blank string (an empty form
    field) or an empty list or tuple."""
    return value is None or (isinstance(value, (str, list, tuple)) and not (value.strip() if isinstance(value, str) else value))


def coerce(cls, params: dict) -> dict:
    """``params``, a model entry's arguments, as ``cls``'s constructor takes them.

    Each value is coerced by its argument's annotation (``coerce_value``);
    None and blank strings are left out, and names the constructor does not
    take are dropped, unless it takes ``**kwargs``.

    Raises:
        ValueError: a value cannot be read as its annotation asks; the
            message starts with the argument's name.
    """
    signature = inspect.signature(cls.__init__)
    try:
        hints = get_type_hints(cls.__init__)
    except Exception:
        hints = {}
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    arguments = {}
    for name, value in params.items():
        if blank(value) or name == "self":
            continue
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.kind is inspect.Parameter.VAR_KEYWORD:
            if takes_any:
                arguments[name] = value
            continue
        try:
            arguments[name] = coerce_value(value, hints.get(name, parameter.annotation))
        except ValueError as e:
            raise ValueError(f"{name} {e}") from None
    return arguments


# --- what the constructors normalize their arguments with --------------------------


def numbers(value, what: str = "values") -> list:
    """``value`` as a list of numbers, ints kept ints and 5.24 kept 5.24: from
    one number, a sequence of them, or text ("8", "16,8,8", "(16, 8, 8)").

    Raises:
        ValueError: something in it is not a number; the message names ``what``.
    """
    if isinstance(value, str):
        value = split(value)
    elif not _is_sequence(value):
        value = [value]
    try:
        return [number(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be numbers, not {value!r}") from None


def voxel_size(value, what: str = "voxel_size"):
    """A voxel size, nm per voxel along z, y and x, as a tuple of three
    numbers: from one number (every axis's), a sequence of three, or text;
    ints kept ints (``int()`` made a 5.24 nm voxel 5). None stays None.

    Raises:
        ValueError: not numbers, not one or three of them, or not positive;
            the message names ``what``.
    """
    if value is None:
        return None
    sizes_nm = numbers(value, what)
    if len(sizes_nm) == 1:
        sizes_nm = sizes_nm * 3
    if len(sizes_nm) != 3 or any(s <= 0 for s in sizes_nm):
        raise ValueError(f"{what} must be one positive number or three (z, y, x), not {value!r}")
    return tuple(sizes_nm)


def sizes(value, *, min=0, lengths=None, what: str = "sizes"):
    """None, or ``value`` (one number, "148", "20,256,256" or a sequence) as a tuple of whole numbers.

    One number stays one (``per_axis`` widens it), so that ``as_given``
    writes it back as it was given. Each must be at least ``min``, and when
    ``lengths`` is given, their count must be one of them.

    Raises:
        ValueError: not numbers, too few or too many, or too small; the
            message names ``what``.
    """
    if value is None:
        return None
    counts = tuple(int(v) for v in numbers(value, what))
    if lengths is not None and len(counts) not in lengths:
        wanted = " or ".join(str(n) for n in lengths)
        raise ValueError(f"{what} must be {wanted} whole numbers, got {list(counts)}")
    if any(s < min for s in counts):
        raise ValueError(f"{what} must be at least {min}, got {list(counts)}")
    return counts


def per_axis(counts, n: int, what: str):
    """``counts`` (``sizes``'s) for each of ``n`` axes: one number is every axis's; None stays None.

    Raises:
        ValueError: neither one number nor ``n``.
    """
    if counts is None:
        return None
    if len(counts) == 1:
        return tuple(counts) * n
    if len(counts) != n:
        raise ValueError(f"{what} needs one number or {n} (one per space axis of the model), got {list(counts)}")
    return tuple(counts)


def as_given(counts):
    """``counts`` (``sizes``'s) as a model entry writes them: one number as itself, more as a list."""
    return counts[0] if len(counts) == 1 else list(counts)


def names(value):
    """None, or channel names as a list of strings, from a sequence or the
    "mito, er" the command line and the model form send."""
    if value is None:
        return None
    if isinstance(value, str):
        value = split(value)
    return [str(v) for v in value]
