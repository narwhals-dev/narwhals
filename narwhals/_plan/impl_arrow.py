"""Translating `ExprIR` nodes for pyarrow.

Acting like a trimmed down, native-only `CompliantExpr`, `CompliantSeries`, etc.
"""

from __future__ import annotations

import typing as t
from functools import singledispatch

from narwhals._plan import expr
from narwhals._plan.literal import is_literal_scalar

if t.TYPE_CHECKING:
    import pyarrow as pa
    from typing_extensions import TypeAlias

    from narwhals._plan.common import ExprIR
    from narwhals._plan.dummy import DummySeries
    from narwhals.typing import NonNestedLiteral

    NativeFrame: TypeAlias = pa.Table
    NativeSeries: TypeAlias = pa.ChunkedArray[t.Any]
    Evaluated: TypeAlias = t.Sequence[NativeSeries]


# TODO @dangotbanned: Update to operate on the output of `expr_expansion` or `expr_rewrites`
# No longer need: `Alias`, `Columns`, `Nth`, `All`, `Exclude`, `IndexColumns`, `RootSelector`, `BinarySelector`, `RenameAlias`, `KeepName`
@singledispatch
def evaluate(node: ExprIR, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.Column)
def col(node: expr.Column, frame: NativeFrame) -> Evaluated:
    return [frame.column(node.name)]


# TODO @dangotbanned: Remove after updating tests
@evaluate.register(expr.Columns)
def cols(node: expr.Columns, frame: NativeFrame) -> Evaluated:
    return frame.select(list(node.names)).columns


@evaluate.register(expr.Literal)
def lit(
    node: expr.Literal[NonNestedLiteral] | expr.Literal[DummySeries[NativeSeries]],
    frame: NativeFrame,
) -> Evaluated:
    import pyarrow as pa

    if is_literal_scalar(node):
        lit: t.Any = pa.scalar
        array = pa.repeat(lit(node.unwrap()), len(frame))
        return [pa.chunked_array([array])]
    return [node.unwrap().to_native()]


@evaluate.register(expr.Len)
def len_(node: expr.Len, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.Cast)
def cast_(node: expr.Cast, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.Ternary)
def ternary(node: expr.Ternary, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.Agg)
def agg(node: expr.Agg, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.OrderableAgg)
def orderable_agg(node: expr.OrderableAgg, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.BinaryExpr)
def binary_expr(node: expr.BinaryExpr, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.FunctionExpr)
def function_expr(node: expr.FunctionExpr[t.Any], frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.RollingExpr)
def rolling_expr(node: expr.RollingExpr[t.Any], frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.WindowExpr)
def window_expr(node: expr.WindowExpr, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.Sort)
def sort(node: expr.Sort, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.SortBy)
def sort_by(node: expr.SortBy, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.Filter)
def filter_(node: expr.Filter, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.AnonymousExpr)
def anonymous_expr(node: expr.AnonymousExpr, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))
