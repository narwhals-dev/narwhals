from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from narwhals._plan.expressions import aggregation as agg, functions as F
from narwhals._plan.expressions.expr import Col, LenStar
from narwhals.typing import RankMethod, RollingInterpolationMethod

if TYPE_CHECKING:
    from collections.abc import Mapping

    from altair.vegalite.v6.schema._typing import AggregateOp_T, WindowOnlyOp_T

    from narwhals._plan import _function, expressions as ir


Ddof: TypeAlias = Literal[0, 1]
"""Only `{var,std}(ddof={0,1})` is supported."""

Quantile: TypeAlias = tuple[RollingInterpolationMethod, float]
"""Only `quantile({0.25,0.75}, "linear")` is supported."""

SUPPORTED_BY_POLARS: tuple[AggregateOp_T | WindowOnlyOp_T, ...] = ("product", "nth_value")

UNSUPPORTED: tuple[AggregateOp_T | WindowOnlyOp_T, ...] = (
    *SUPPORTED_BY_POLARS,
    "ci0",
    "ci1",
    "stderr",
    "exponential",  # not documented
    "exponentialb",  # not documented
    # window_only
    # TODO @dangotbanned: Search for these on polars tracker
    "cume_dist",
    "percent_rank",
    "ntile",
)

# NOTE: "auto-implode" is the only option available
# need to add an actual `Implode` aggregation, but keep this too
Implode: Final = Col

AGG_EXPR: Mapping[type[ir.AggExpr | ir.ExprIR], AggregateOp_T] = {
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

# NOTE: Only supported when `dd`
AGG_EXPR_VAR_STD: Mapping[tuple[type[agg.Std | agg.Var], Ddof], AggregateOp_T] = {
    (agg.Std, 0): "stdevp",
    (agg.Std, 1): "stdev",
    (agg.Var, 0): "variancep",
    (agg.Var, 1): "variance",
}
AGG_EXPR_QUANTILE: Mapping[Quantile, AggregateOp_T] = {
    ("linear", 0.25): "q1",
    ("linear", 0.75): "q3",
}
AGG_FUNC: Mapping[type[_function.Aggregation], AggregateOp_T] = {F.NullCount: "missing"}


WINDOW_EXPR: Mapping[type[ir.AggExpr], WindowOnlyOp_T] = {
    agg.First: "first_value",
    agg.Last: "last_value",
}

RANK_METHOD_WINDOW: Mapping[RankMethod, WindowOnlyOp_T] = {
    "ordinal": "row_number",
    "dense": "dense_rank",
    "min": "rank",  # not sure if this is equivalent
}


def shift_to_window_op(f: F.Shift, /) -> tuple[Literal["lag", "lead"], int]:
    n = f.n
    return ("lead", n) if n >= 1 else ("lag", abs(n))
