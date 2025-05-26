"""Translating `ExprIR` nodes for pyarrow.

Acting like a trimmed down, native-only `CompliantExpr`, `CompliantSeries`, etc.
"""

from __future__ import annotations

import typing as t
from functools import singledispatch

from narwhals._plan import expr
from narwhals._plan.literal import is_scalar_literal, is_series_literal

if t.TYPE_CHECKING:
    import pyarrow as pa
    from typing_extensions import TypeAlias

    from narwhals._plan.common import ExprIR

    NativeFrame: TypeAlias = pa.Table
    NativeSeries: TypeAlias = pa.ChunkedArray[t.Any]
    Evaluated: TypeAlias = t.Sequence[NativeSeries]


@singledispatch
def evaluate(node: ExprIR, frame: NativeFrame) -> Evaluated:
    raise NotImplementedError(type(node))


@evaluate.register(expr.Column)
def col(node: expr.Column, frame: NativeFrame) -> Evaluated:
    return [frame.column(node.name)]


@evaluate.register(expr.Columns)
def cols(node: expr.Columns, frame: NativeFrame) -> Evaluated:
    return frame.select(list(node.names)).columns


@evaluate.register(expr.Literal)
def lit(node: expr.Literal, frame: NativeFrame) -> Evaluated:  # noqa: ARG001
    import pyarrow as pa

    if is_scalar_literal(node.value):
        return [pa.chunked_array([node.value.unwrap()])]
    elif is_series_literal(node.value):
        ca = node.value.unwrap().to_native()
        return [t.cast("NativeSeries", ca)]
    else:
        raise NotImplementedError(type(node.value))
