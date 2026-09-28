"""Lambda normalizer/postprocessor expressions may only be numpy math on x.

They arrive in layer URLs, so anything that let an expression reach Python
builtins, dunder attributes or unbounded work would let a URL run code or
stall an inference server.
"""

import numpy as np
import pytest

from cellmap_flow.norm.input_normalize import LambdaNormalizer
from cellmap_flow.post.postprocessors import LambdaPostprocessor
from cellmap_flow.utils.safe_expression import compile_expression

X = np.array([[-1.0, 0.25], [0.75, 2.0]], dtype=np.float32)


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("x*2-1", X * 2 - 1),
        ("np.clip(x, 0, 1)", np.clip(X, 0, 1)),
        ("(x > 0.5).astype(np.uint8)", (X > 0.5).astype(np.uint8)),
        ("x.astype('uint8')", X.astype("uint8")),
        ("np.where(x > 0, x, 0)", np.where(X > 0, X, 0)),
        ("x[..., 0]", X[..., 0]),
        ("x ** 2 + -x", X**2 - X),
        ("abs(x) / np.max(abs(x))", np.abs(X) / np.abs(X).max()),
        ("x if x.ndim == 2 else x * 0", X),
        ("np.clip(x, a_min=0, a_max=None)", np.clip(X, 0, None)),
    ],
)
def test_numpy_math_on_x_is_allowed(expression, expected):
    np.testing.assert_array_equal(compile_expression(expression)(X), expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "open('/etc/passwd').read()",
        "x.__class__.__mro__",
        "().__class__.__bases__[0].__subclasses__()",
        "np.load('/tmp/anything.npy', allow_pickle=True)",
        "getattr(x, 'shape')",
        "exec('1')",
        "(lambda: 1)()",
        "[y for y in x]",
        "{**{}}",
        "x ** 10**10",
        "x ** x",
        "x.astype('U1000')",
        "x = 1",
        "x" * 600,
    ],
)
def test_anything_else_is_rejected(expression):
    with pytest.raises(ValueError):
        compile_expression(expression)


@pytest.mark.parametrize("cls", [LambdaNormalizer, LambdaPostprocessor])
def test_lambda_ops_refuse_unsafe_expressions_at_construction(cls):
    with pytest.raises(ValueError, match="not allowed"):
        cls("__import__('os').getcwd()")


@pytest.mark.parametrize("cls", [LambdaNormalizer, LambdaPostprocessor])
def test_lambda_ops_still_evaluate_ordinary_expressions(cls):
    np.testing.assert_allclose(cls("x*2-1")(X), X * 2 - 1)
