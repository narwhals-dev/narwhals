"""Translating `ExprIR` nodes for pyarrow."""

from __future__ import annotations

import typing as t

# ruff: noqa: ARG001
from functools import singledispatch
from itertools import repeat

from narwhals._plan import aggregation as agg, expr
from narwhals._plan.arrow.series import ArrowSeries
from narwhals._plan.literal import is_literal_scalar

if t.TYPE_CHECKING:
    from typing_extensions import TypeAlias, TypeIs

    from narwhals._arrow.typing import (  # type: ignore[attr-defined]
        ChunkedArrayAny,
        Order,
        ScalarAny,
    )
    from narwhals._plan.arrow.dataframe import ArrowDataFrame
    from narwhals._plan.common import ExprIR, NamedIR
    from narwhals._plan.dummy import DummySeries
    from narwhals._plan.protocols import SupportsBroadcast
    from narwhals.typing import NonNestedLiteral, PythonLiteral


UnaryFn: TypeAlias = "t.Callable[[ChunkedArrayAny], ScalarAny]"


def is_scalar(obj: t.Any) -> TypeIs[ScalarAny]:
    import pyarrow as pa  # ignore-banned-import

    return isinstance(obj, pa.Scalar)


def evaluate(
    node: NamedIR[ExprIR], frame: ArrowDataFrame
) -> SupportsBroadcast[ArrowSeries]:
    result = _evaluate_inner(node.expr, frame)
    if is_scalar(result):
        return frame._lit.from_scalar(result, node.name)
    return frame._expr.from_native(result, node.name)


# NOTE: Should mean we produce 1x CompliantSeries for the entire expression
# Multi-output have already been separated
# No intermediate CompliantSeries need to be created, just assign a name to the final one
@singledispatch
def _evaluate_inner(node: ExprIR, frame: ArrowDataFrame) -> ChunkedArrayAny:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.Column)
def col(node: expr.Column, frame: ArrowDataFrame) -> ChunkedArrayAny:
    return frame.native.column(node.name)


# NOTE: Using a very naïve approach to broadcasting **for now**
# - We already have something that works in main
# - Another approach would be to keep everything wrapped (or aggregated into)  `expr.Literal`
def _lit_native(
    value: PythonLiteral | ScalarAny, frame: ArrowDataFrame
) -> ChunkedArrayAny:
    """Will need to support returning a native scalar as well."""
    import pyarrow as pa  # ignore-banned-import

    from narwhals._arrow.utils import chunked_array

    lit: t.Any = pa.scalar
    scalar: t.Any = value if isinstance(value, pa.Scalar) else lit(value)
    array = pa.repeat(scalar, len(frame))
    return chunked_array(array)


@_evaluate_inner.register(expr.Literal)
def lit_(
    node: expr.Literal[NonNestedLiteral] | expr.Literal[DummySeries[ChunkedArrayAny]],
    frame: ArrowDataFrame,
) -> ChunkedArrayAny:
    if is_literal_scalar(node):
        return _lit_native(node.unwrap(), frame)
    return node.unwrap().to_native()


@_evaluate_inner.register(expr.Cast)
def cast_(node: expr.Cast, frame: ArrowDataFrame) -> ChunkedArrayAny:
    from narwhals._arrow.utils import narwhals_to_native_dtype

    data_type = narwhals_to_native_dtype(node.dtype, frame.version)
    return _evaluate_inner(node.expr, frame).cast(data_type)


@_evaluate_inner.register(expr.Sort)
def sort(node: expr.Sort, frame: ArrowDataFrame) -> ChunkedArrayAny:
    import pyarrow.compute as pc

    native = _evaluate_inner(node.expr, frame)
    sorted_indices = pc.array_sort_indices(native, options=node.options.to_arrow())
    return native.take(sorted_indices)


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


@_evaluate_inner.register(expr.Filter)
def filter_(node: expr.Filter, frame: ArrowDataFrame) -> ChunkedArrayAny:
    return _evaluate_inner(node.expr, frame).filter(_evaluate_inner(node.by, frame))


@_evaluate_inner.register(expr.Len)
def len_(node: expr.Len, frame: ArrowDataFrame) -> ChunkedArrayAny:
    return _lit_native(len(frame), frame)


@_evaluate_inner.register(expr.Ternary)
def ternary(node: expr.Ternary, frame: ArrowDataFrame) -> ChunkedArrayAny:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(agg.Last)
@_evaluate_inner.register(agg.First)
def agg_first_last(node: agg.First | agg.Last, frame: ArrowDataFrame) -> ChunkedArrayAny:
    native = _evaluate_inner(node.expr, frame)
    if height := len(native):
        result = native[height - 1 if isinstance(node, agg.Last) else 0]
    else:
        result = None
    return _lit_native(result, frame)


@_evaluate_inner.register(agg.ArgMax)
@_evaluate_inner.register(agg.ArgMin)
def agg_arg_min_max(
    node: agg.ArgMin | agg.ArgMax, frame: ArrowDataFrame
) -> ChunkedArrayAny:
    import pyarrow.compute as pc

    native = _evaluate_inner(node.expr, frame)
    fn = pc.min if isinstance(node, agg.ArgMin) else pc.max
    result = pc.index(native, fn(native))
    return _lit_native(result, frame)


@_evaluate_inner.register(agg.Sum)
def agg_sum(node: agg.Sum, frame: ArrowDataFrame) -> ChunkedArrayAny:
    import pyarrow.compute as pc

    result = pc.sum(_evaluate_inner(node.expr, frame), min_count=0)
    return _lit_native(result, frame)


@_evaluate_inner.register(agg.NUnique)
def agg_n_unique(node: agg.NUnique, frame: ArrowDataFrame) -> ChunkedArrayAny:
    import pyarrow.compute as pc

    result = pc.count(_evaluate_inner(node.expr, frame).unique(), mode="all")
    return _lit_native(result, frame)


@_evaluate_inner.register(agg.Var)
@_evaluate_inner.register(agg.Std)
def agg_std_var(node: agg.Std | agg.Var, frame: ArrowDataFrame) -> ChunkedArrayAny:
    import pyarrow.compute as pc

    fn = pc.stddev if isinstance(node, agg.Std) else pc.variance
    result = fn(_evaluate_inner(node.expr, frame), ddof=node.ddof)
    return _lit_native(result, frame)


@_evaluate_inner.register(agg.Quantile)
def agg_quantile(node: agg.Quantile, frame: ArrowDataFrame) -> ChunkedArrayAny:
    import pyarrow.compute as pc

    result = pc.quantile(
        _evaluate_inner(node.expr, frame),
        q=node.quantile,
        interpolation=node.interpolation,
    )[0]
    return _lit_native(result, frame)


@_evaluate_inner.register(expr.Agg)
def agg_expr(node: expr.Agg, frame: ArrowDataFrame) -> ChunkedArrayAny:
    import pyarrow.compute as pc

    mapping: dict[type[expr.Agg], UnaryFn] = {
        agg.Count: pc.count,
        agg.Max: pc.max,
        agg.Mean: pc.mean,
        agg.Median: pc.approximate_median,
        agg.Min: pc.min,
    }
    if fn := mapping.get(type(node)):
        result = fn(_evaluate_inner(node.expr, frame))
        return _lit_native(result, frame)
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
