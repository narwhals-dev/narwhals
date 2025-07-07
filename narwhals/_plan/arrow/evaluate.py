"""TODO: Move all the impls `ArrowExpr`/`ArrowScalar`, then delete."""

from __future__ import annotations

import typing as t
from functools import singledispatch
from itertools import repeat

from narwhals._plan import expr
from narwhals._plan.arrow.series import ArrowSeries

if t.TYPE_CHECKING:
    from typing_extensions import TypeAlias, TypeIs

    from narwhals._arrow.typing import (  # type: ignore[attr-defined]
        ChunkedArrayAny,
        Order,
        ScalarAny,
    )
    from narwhals._plan.arrow.dataframe import ArrowDataFrame
    from narwhals._plan.common import ExprIR, NamedIR
    from narwhals._plan.protocols import EagerBroadcast


UnaryFn: TypeAlias = "t.Callable[[ChunkedArrayAny], ScalarAny]"


def is_scalar(obj: t.Any) -> TypeIs[ScalarAny]:
    import pyarrow as pa  # ignore-banned-import

    return isinstance(obj, pa.Scalar)


def evaluate(node: NamedIR[ExprIR], frame: ArrowDataFrame) -> EagerBroadcast[ArrowSeries]:
    result = _evaluate_inner(node.expr, frame)
    if is_scalar(result):
        return frame.__narwhals_namespace__()._scalar.from_native(result, node.name)
    return frame.__narwhals_namespace__()._expr.from_native(result, node.name)


# NOTE: Should mean we produce 1x CompliantSeries for the entire expression
# Multi-output have already been separated
# No intermediate CompliantSeries need to be created, just assign a name to the final one
@singledispatch
def _evaluate_inner(node: ExprIR, frame: ArrowDataFrame) -> ChunkedArrayAny:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.Column)
def col(node: expr.Column, frame: ArrowDataFrame) -> ChunkedArrayAny:
    return frame.native.column(node.name)


@_evaluate_inner.register(expr.SortBy)
def sort_by(node: expr.SortBy, frame: ArrowDataFrame) -> ChunkedArrayAny:
    opts = node.options
    if len(opts.nulls_last) != 1:
        msg = f"pyarrow doesn't support multiple values for `nulls_last`, got: {opts.nulls_last!r}"
        raise NotImplementedError(msg)
    placement = "at_end" if opts.nulls_last[0] else "at_start"
    from_native = ArrowSeries.from_native
    by = (
        from_native(_evaluate_inner(e, frame), str(idx)) for idx, e in enumerate(node.by)
    )
    df = frame.from_series(from_native(_evaluate_inner(node.expr, frame), "<TEMP>"), *by)
    names = df.columns[1:]
    if len(opts.descending) == 1:
        descending: t.Iterable[bool] = repeat(opts.descending[0], len(names))
    else:
        descending = opts.descending
    sorting: list[tuple[str, Order]] = [
        (key, "descending" if desc else "ascending")
        for key, desc in zip(names, descending)
    ]
    return df.native.sort_by(sorting, null_placement=placement).column(0)


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
