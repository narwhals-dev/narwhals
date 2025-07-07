"""TODO: Define remaining nodes in `Dispatch` protocol.

- `Ternary`
- `BinaryExpr`
- `FunctionExpr`
- `RollingExpr`
- `WindowExpr`
- `OrderedWindowExpr`
- `AnonymousExpr`
"""

from __future__ import annotations

import typing as t
from functools import singledispatch

from narwhals._plan import expr

if t.TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._arrow.typing import ChunkedArrayAny, ScalarAny
    from narwhals._plan.arrow.dataframe import ArrowDataFrame
    from narwhals._plan.arrow.series import ArrowSeries
    from narwhals._plan.common import ExprIR, NamedIR
    from narwhals._plan.protocols import EagerBroadcast


def is_scalar(obj: t.Any) -> TypeIs[ScalarAny]:
    import pyarrow as pa  # ignore-banned-import

    return isinstance(obj, pa.Scalar)


def evaluate(node: NamedIR[ExprIR], frame: ArrowDataFrame) -> EagerBroadcast[ArrowSeries]:
    result = _evaluate_inner(node.expr, frame)
    if is_scalar(result):
        return frame.__narwhals_namespace__()._scalar.from_native(result, node.name)
    return frame.__narwhals_namespace__()._expr.from_native(result, node.name)


@singledispatch
def _evaluate_inner(node: ExprIR, frame: ArrowDataFrame) -> ChunkedArrayAny:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.Ternary)
def ternary(node: expr.Ternary, frame: ArrowDataFrame) -> ChunkedArrayAny:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.BinaryExpr)
def binary_expr(node: expr.BinaryExpr, frame: ArrowDataFrame) -> ChunkedArrayAny:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.FunctionExpr)
def function_expr(
    node: expr.FunctionExpr[t.Any], frame: ArrowDataFrame
) -> ChunkedArrayAny:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.RollingExpr)
def rolling_expr(node: expr.RollingExpr[t.Any], frame: ArrowDataFrame) -> ChunkedArrayAny:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.WindowExpr)
def window_expr(node: expr.WindowExpr, frame: ArrowDataFrame) -> ChunkedArrayAny:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.AnonymousExpr)
def anonymous_expr(node: expr.AnonymousExpr, frame: ArrowDataFrame) -> ChunkedArrayAny:
    raise NotImplementedError(type(node))
