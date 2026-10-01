"""Evaluate the small array expressions behind LambdaNormalizer and LambdaPostprocessor.

These expressions arrive inside neuroglancer layer URLs and dashboard requests,
and inference servers listen on every interface, so evaluating them with a bare
eval() let anyone who could reach a server run code on it. An expression is now
parsed once and every node is checked against a whitelist before anything runs:

- the array ``x``, numbers, and numeric dtype names as strings (``x.astype("uint8")``);
- arithmetic, comparison and boolean operators, and ``a if cond else b``;
- indexing and slicing (``x[..., 0]``);
- ``np.<name>`` for the numpy functions, dtypes and constants listed below;
- a few array methods and attributes (``x.astype``, ``x.clip``, ``x.shape``, ...);
- ``abs``.

Anything else (other names, other attributes, dunders, lambdas, comprehensions,
``**kwargs``) raises ValueError. So does what would let a short expression
allocate without bound: exponents must be literal numbers of modest size (no
``x ** 10**10``), a list or tuple cannot be repeated (no ``[x] * 10**5``), keyword
arguments are only those in ``KEYWORDS`` (no ``shape=``), and the ``*_like``
functions take no positional shape either.

That bounds what a whitelisted call can be asked for, not what arithmetic can
produce: broadcasting arrays given new axes (``x[..., None] * x[None]``) still
multiplies a chunk's size, so the whitelist is not a memory limit.
"""

import ast
import types
from typing import Callable

import numpy as np

NUMPY_FUNCTIONS = frozenset(
    {
        "abs", "absolute", "amax", "amin", "arccos", "arcsin", "arctan", "argmax",
        "argmin", "ceil", "clip", "concatenate", "cos", "exp", "expand_dims", "floor",
        "full_like", "isfinite", "isnan", "log", "log10", "log1p", "log2",
        "logical_and", "logical_not", "logical_or", "max", "maximum", "mean", "min",
        "minimum", "nan_to_num", "ones_like", "percentile", "power", "rint", "round",
        "sign", "sin", "sqrt", "square", "squeeze", "stack", "std", "sum", "tanh",
        "where", "zeros_like",
    }
)
NUMPY_DTYPES = frozenset(
    {
        "bool_", "float16", "float32", "float64", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
    }
)
NUMPY_CONSTANTS = frozenset({"e", "inf", "nan", "pi"})
ARRAY_METHODS = frozenset(
    {"astype", "clip", "max", "mean", "min", "round", "squeeze", "std", "sum"}
)
ARRAY_ATTRIBUTES = frozenset({"dtype", "ndim", "shape"})

# Strings are only for dtype arguments; a free-form string like "U1000" would
# let x.astype() multiply a chunk's memory by thousands.
DTYPE_STRINGS = NUMPY_DTYPES | {"bool", "float", "int"}

# Every keyword argument a whitelisted call may be given. Not ``shape``,
# which would let np.full_like allocate any size.
KEYWORDS = frozenset({"axis", "keepdims", "dtype", "a_min", "a_max"})
# The positional arguments a ``*_like`` function may take: the array, its
# fill value, its dtype. One more would be ``order``, and then ``shape``.
MAX_POSITIONAL = {"full_like": 3, "ones_like": 2, "zeros_like": 2}

MAX_LENGTH = 500
MAX_EXPONENT = 64

_NP = types.SimpleNamespace(
    **{name: getattr(np, name) for name in NUMPY_FUNCTIONS | NUMPY_DTYPES | NUMPY_CONSTANTS}
)
_NAMES = {"x", "np", "abs"}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Load, ast.keyword,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.Invert, ast.Not, ast.UAdd, ast.USub,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def compile_expression(expression: str) -> Callable[[np.ndarray], np.ndarray]:
    """Check ``expression`` against the whitelist and return a function of ``x``.

    Raises ValueError, naming the offending construct, if the expression is
    not allowed.
    """
    if not isinstance(expression, str):
        raise ValueError(f"Expression must be a string, got {type(expression).__name__}")
    if len(expression) > MAX_LENGTH:
        raise ValueError(f"Expression is longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression {expression!r}: {e.msg}") from None
    for node in ast.walk(tree):
        _check(node, expression)
    code = compile(tree, "<expression>", "eval")

    def evaluate(x):
        return eval(code, {"__builtins__": {}}, {"x": x, "np": _NP, "abs": abs})

    return evaluate


def _reject(expression, what):
    raise ValueError(f"Expression {expression!r} is not allowed: {what}")


def _check(node, expression):
    if isinstance(node, _ALLOWED_NODES):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            try:
                exponent = ast.literal_eval(node.right)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                _reject(expression, "exponents must be literal numbers")
            if not isinstance(exponent, (int, float)) or abs(exponent) > MAX_EXPONENT:
                _reject(expression, f"exponents must be numbers with |n| <= {MAX_EXPONENT}")
        if isinstance(node, ast.BinOp) and any(
            isinstance(side, (ast.List, ast.Tuple)) for side in (node.left, node.right)
        ):
            _reject(expression, "arithmetic on a list or tuple, such as repeating it")
        if isinstance(node, ast.keyword) and node.arg not in KEYWORDS:
            _reject(expression, f"keyword {node.arg}= (only {', '.join(sorted(KEYWORDS))})")
        return
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, str):
            if value in DTYPE_STRINGS:
                return
            _reject(expression, f"string {value!r} (strings may only name numeric dtypes)")
        if value is Ellipsis or value is None or isinstance(value, (bool, int, float)):
            return
        _reject(expression, f"constant {value!r}")
    if isinstance(node, ast.Name):
        if node.id in _NAMES:
            return
        _reject(expression, f"name {node.id!r} (only x, np and abs are available)")
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "np":
            if node.attr in NUMPY_FUNCTIONS | NUMPY_DTYPES | NUMPY_CONSTANTS:
                return
            _reject(expression, f"np.{node.attr}")
        if node.attr in ARRAY_METHODS | ARRAY_ATTRIBUTES:
            return
        _reject(expression, f"attribute .{node.attr}")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) or (
            isinstance(node.func, ast.Name) and node.func.id == "abs"
        ):
            limit = MAX_POSITIONAL.get(getattr(node.func, "attr", None))
            if limit is not None and len(node.args) > limit:
                _reject(expression, f"more than {limit} positional arguments to {node.func.attr}")
            return  # the callee and the arguments are checked as their own nodes
        _reject(expression, "only np functions, array methods and abs can be called")
    _reject(expression, type(node).__name__)
