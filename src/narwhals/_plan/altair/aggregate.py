from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, overload

try:
    import altair as alt
except ImportError as err:
    msg = "`altair` is required to convert `ExprIR`s to transformations."
    raise ModuleNotFoundError(msg) from err
from narwhals._plan import expressions as ir
from narwhals._plan.altair.exceptions import Target, unsupported_error
from narwhals._plan.altair.parse import parse_into_named_exprs
from narwhals._plan.altair.typing import (
    AggOrWindow,
    AggregateOp,
    IntoExprColumn,
    OutputName,
    WindowOp,
)
from narwhals._plan.expressions import aggregation as agg, functions as F
from narwhals._plan.expressions.expr import Col

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._plan import _function as _f
    from narwhals._plan.altair import typing as alt_t
    from narwhals.typing import RankMethod


Incomplete: TypeAlias = Any

WindowParam: TypeAlias = tuple[AggOrWindow, alt.typing.Optional[float]]
"""`(op, param)` in a `WindowFieldDef`."""


SUPPORTED_BY_POLARS: tuple[AggOrWindow, ...] = ("product", "nth_value")
UNSUPPORTED: tuple[AggOrWindow, ...] = (
    *SUPPORTED_BY_POLARS,
    "ci0",
    "ci1",
    "stderr",
    "exponential",  # not documented
    "exponentialb",  # not documented
    # window_only
    "cume_dist",
    "percent_rank",
    "ntile",
)

# NOTE: "auto-implode" is the only option available
# need to add an actual `Implode` aggregation, but keep this too
Implode: Final = Col


_AggOptions: TypeAlias = tuple[Incomplete, ...]
"""Arguments from an aggregation expression that are supported."""

_AggExprKey: TypeAlias = tuple[type[ir.AggExpr], _AggOptions]

AGG_EXPR: Mapping[_AggExprKey, AggregateOp] = {
    (agg.Count, ()): "valid",
    (agg.Len, ()): "count",  # > returns the same value regardless of ... field.
    (agg.Max, ()): "max",
    (agg.Mean, ()): "mean",
    (agg.Median, ()): "median",
    (agg.Min, ()): "min",
    (agg.NUnique, ()): "distinct",
    (agg.Sum, ()): "sum",
    (agg.Std, (0,)): "stdevp",
    (agg.Std, (1,)): "stdev",
    (agg.Var, (0,)): "variancep",
    (agg.Var, (1,)): "variance",
    (agg.Quantile, ("linear", 0.25)): "q1",
    (agg.Quantile, ("linear", 0.75)): "q3",
}
"""Mapping from supported aggregations to their `AggregateOp` equivalent."""


AGG_FUNC: Mapping[type[_f.Aggregation], AggregateOp] = {F.NullCount: "missing"}


WINDOW_EXPR: Mapping[type[ir.AggExpr], WindowOp] = {
    agg.First: "first_value",
    agg.Last: "last_value",
}


# NOTE: Need to enforce `alt.WindowTransform(frame=(None,0))`
WINDOW_FUNC_CUM: Mapping[type[F.CumAgg], AggregateOp] = {
    F.CumProd: "product",
    F.CumMin: "min",
    F.CumMax: "max",
    F.CumSum: "sum",
    F.CumCount: "count",
}


# NOTE: Need to convert `RollingOptions` -> frame=(...)
WINDOW_FUNC_ROLLING: Mapping[type[F.RollingWindow], AggregateOp] = {
    F.RollingMean: "mean",
    F.RollingSum: "sum",
}
# NOTE: Need to convert `RollingOptions` -> frame=(...)
WINDOW_FUNC_ROLLING_VAR_STD: Mapping[
    tuple[type[F.RollingStd | F.RollingVar], int], AggregateOp
] = {
    (F.RollingStd, 0): "stdevp",
    (F.RollingStd, 1): "stdev",
    (F.RollingVar, 0): "variancep",
    (F.RollingVar, 1): "variance",
}


RANK_METHOD_WINDOW: Mapping[RankMethod, WindowOp] = {
    "ordinal": "row_number",
    "dense": "dense_rank",
    "min": "rank",  # not sure if this is equivalent
}


@functools.singledispatch
def _function_window(f: ir.Function, /) -> WindowParam | None:  # noqa: ARG001
    return None


for _tp in AGG_FUNC:
    _function_window.register(_tp, lambda f: (AGG_FUNC[type(f)], alt.Undefined))


@_function_window.register(F.Shift)
def shift_to_window_op(f: F.Shift, /) -> WindowParam:
    n = f.n
    return ("lead", n) if n >= 1 else ("lag", abs(n))


@_function_window.register(F.Rank)
def rank_to_window_op(f: F.Rank, /) -> WindowParam | None:
    opts = f.options
    if (op := RANK_METHOD_WINDOW.get(opts.method)) and not opts.descending:
        return op, alt.Undefined
    return None


@overload
def from_agg_expr(expr: ir.AggExpr, context: Literal["window"]) -> alt_t.WindowField: ...
@overload
def from_agg_expr(expr: ir.AggExpr, context: Literal["aggregate"]) -> alt_t.AggField: ...
@overload
def from_agg_expr(expr: ir.AggExpr, context: Literal["encoding"]) -> alt_t.Field: ...
def from_agg_expr(
    expr: ir.AggExpr, context: Target
) -> alt_t.Field | alt_t.AggField | alt_t.WindowField:
    prev = expr.expr
    if isinstance(prev, Col):
        key = _agg_expr_key(expr)
        if (op := AGG_EXPR.get(key)) is None:
            raise unsupported_error(
                expr, context, reason=("non-default" if key[1] != () else None)
            )

        return {"field": prev.name, "op": op}

    raise unsupported_error(expr, context)


@functools.singledispatch
def _agg_expr_key(expr: ir.AggExpr) -> _AggExprKey:
    """Destructure an `AggExpr` into the parts that dictate if we can support it."""
    return type(expr), ()


@_agg_expr_key.register(agg.Quantile)
def _(expr: agg.Quantile) -> _AggExprKey:
    return type(expr), (expr.interpolation, expr.quantile)


@_agg_expr_key.register(agg.Std)
@_agg_expr_key.register(agg.Var)
def _(expr: agg.Std | agg.Var) -> _AggExprKey:
    return type(expr), (expr.ddof,)


# TODO @dangotbanned: Simplify the `AggExpr` branch
# TODO @dangotbanned: Change the shape of it to allow using `over` to split out multiple `alt.WindowTransform`s
def window_field_def(
    alias: OutputName, expr: ir.ExprIR
) -> alt.WindowFieldDef | Incomplete:
    """Try to convert a narwhals expression to part of a `WindowTransform`.

    - by default, everything is cumulative
    - `frame` can be used to define either cumulative or rolling
    """
    op: AggOrWindow
    param: alt.typing.Optional[Any] = alt.Undefined
    kwds = {"as": alias}
    if isinstance(expr, ir.FunctionExpr):
        if (window_op := _function_window(expr.function)) is None:
            raise unsupported_error(expr, "window")
        op, param = window_op
        if not isinstance(expr.args[0], Col):
            raise unsupported_error(expr, "window")
        field = expr.args[0].name

    elif isinstance(expr, ir.AggExpr):
        return alt.WindowFieldDef(**from_agg_expr(expr, "window"), **kwds)

    elif isinstance(expr, Col):
        op = "values"
        field = expr.name
    else:
        raise unsupported_error(expr, "window")
    return alt.WindowFieldDef(op=op, field=field, param=param, **kwds)


def window_transform(
    *exprs: IntoExprColumn, **named_exprs: IntoExprColumn
) -> alt.WindowTransform:
    """Parse into narwhals expressions and translate to a single window transform."""
    parsed = parse_into_named_exprs(*exprs, **named_exprs)
    return alt.WindowTransform([window_field_def(alias, expr) for alias, expr in parsed])


# TODO @dangotbanned: convert `over` -> `WindowTransform
def over_window_transform(expr: ir.Over | ir.OverOrdered) -> alt.WindowTransform:
    msg = (
        f"TODO: convert `over` -> `WindowTransform`, got {expr!r}.\n"
        "alt.WindowTransform(window=expr.expr, frame=(...), groupby=expr.partition_by, sort=alt.SortField(field=order_by[i], order='descending'))"
    )
    raise NotImplementedError(msg)
