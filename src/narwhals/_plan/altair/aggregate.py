from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Final, TypeAlias

try:
    import altair as alt
except ImportError as err:
    msg = "`altair` is required to convert `ExprIR`s to transformations."
    raise ModuleNotFoundError(msg) from err
from narwhals._plan import expressions as ir
from narwhals._plan.altair.exceptions import unsupported_error
from narwhals._plan.altair.parse import parse_into_named_exprs
from narwhals._plan.altair.typing import (
    AggOrWindow,
    AggregateOp,
    IntoExprColumn,
    OutputName,
    WindowOp,
)
from narwhals._plan.expressions import aggregation as agg, functions as F
from narwhals._plan.expressions.expr import Col, LenStar
from narwhals.typing import RankMethod, RollingInterpolationMethod

if TYPE_CHECKING:
    from collections.abc import Mapping

    from _typeshed import Incomplete

    from narwhals._plan import _function as _f


Ddof: TypeAlias = int
"""Only `{var,std}(ddof={0,1})` is supported."""

Quantile: TypeAlias = tuple[RollingInterpolationMethod, float]
"""Only `quantile({0.25,0.75}, "linear")` is supported."""

WindowParam: TypeAlias = tuple[AggOrWindow, Any]
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

AGG_EXPR: Mapping[type[ir.AggExpr | ir.ExprIR], AggregateOp] = {
    agg.ArgMax: "argmax",
    agg.ArgMin: "argmin",
    agg.Count: "valid",
    agg.Len: "count",  # > Note: 'count' operates directly on the input objects and return the same value regardless of the provided field.
    LenStar: "count",
    agg.Max: "max",
    agg.Mean: "mean",
    agg.Median: "median",
    agg.Min: "min",
    agg.NUnique: "distinct",
    agg.Sum: "sum",
    Implode: "values",
}

AGG_EXPR_VAR_STD: Mapping[tuple[type[agg.Std | agg.Var], Ddof], AggregateOp] = {
    (agg.Std, 0): "stdevp",
    (agg.Std, 1): "stdev",
    (agg.Var, 0): "variancep",
    (agg.Var, 1): "variance",
}
AGG_EXPR_QUANTILE: Mapping[Quantile, AggregateOp] = {
    ("linear", 0.25): "q1",
    ("linear", 0.75): "q3",
}
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
    tuple[type[F.RollingStd | F.RollingVar], Ddof], AggregateOp
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


# TODO @dangotbanned: Implement `transform_aggregate` support
# (remember that this should be adaptable for encodings like `alt.PositionFieldDef`)
def aggregated_field_def(
    alias: OutputName,
    expr: ir.ExprIR,  # noqa: ARG001
) -> alt.AggregatedFieldDef | Incomplete:
    op: AggregateOp  # noqa: F842
    kwds = {"as": alias}  # noqa: F841
    raise NotImplementedError("todo")


def agg_field_def_to_position_field_def(
    obj: alt.AggregatedFieldDef,  # noqa: ARG001
) -> alt.PositionFieldDef:
    # NOTE: there's no typing for attribute access
    _remap_fields = {"op": "aggregate", "field": "field", "as": "title"}
    raise NotImplementedError("todo")


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
            raise unsupported_error(expr, "window transform")
        op, param = window_op
        if not isinstance(expr.args[0], ir.Column):
            raise unsupported_error(expr, "window transform")
        field = expr.args[0].name

    elif isinstance(expr, ir.AggExpr):
        if (supported := AGG_EXPR.get(type(expr))) is None:
            match expr:
                case agg.Quantile(interpolation=i, quantile=q) if (
                    i,
                    q,
                ) in AGG_EXPR_QUANTILE:
                    op = AGG_EXPR_QUANTILE[(i, q)]
                case agg.Std(ddof=ddof) | agg.Var(ddof=ddof) if ddof in {0, 1}:
                    op = AGG_EXPR_VAR_STD[(type(expr), ddof)]
                case _:
                    raise unsupported_error(expr, "window transform")
        else:
            op = supported
        if not isinstance(expr.expr, ir.Column):
            raise unsupported_error(expr, "window transform")
        field = expr.expr.name

    elif isinstance(expr, ir.Column):
        op = "values"
        field = expr.name
    else:
        raise unsupported_error(expr, "window transform")
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
