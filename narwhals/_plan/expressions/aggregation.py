"""Aggregation expressions.

## Implementation Notes
Adapted from [`dsl::expr::AggExpr`], extended with `Arg{Min,Max}`.

They're similar to `Function` [upstream] - but also appear in [`GroupByMethod`] like the rest of `AggExpr`.

Note that there are a few other aggregating functions that do not fit this definition:

    All, Any, NullCount, ModeAny, Kurtosis, Skew

In the future it might make sense to deviate from polars and move them here ([(#3353 (comment))]).

[`dsl::expr::AggExpr`]: https://github.com/pola-rs/polars/blob/2b825ad7933e4b7ca88556f67f4323caaaa40644/crates/polars-plan/src/dsl/expr/mod.rs#L23-L61
[upstream]: https://github.com/pola-rs/polars/blob/2b825ad7933e4b7ca88556f67f4323caaaa40644/crates/polars-plan/src/dsl/function_expr/mod.rs#L165-L166
[`GroupByMethod`]: https://github.com/pola-rs/polars/blob/2b825ad7933e4b7ca88556f67f4323caaaa40644/crates/polars-core/src/frame/group_by/mod.rs#L860-L882
[(#3353 (comment))]: https://github.com/narwhals-dev/narwhals/pull/3353#discussion_r2622679274
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._expr_ir import ExprIR
from narwhals._plan._nodes import node
from narwhals._plan.exceptions import agg_scalar_error

if TYPE_CHECKING:
    from narwhals.typing import RollingInterpolationMethod

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
map_first = ResolveDType.expr_ir.map_first
same_dtype = ResolveDType.expr_ir.same_dtype


class AggExpr(ExprIR):
    """An aggregation reduces an expression to one that is always scalar.

    >>> import narwhals._plan as nw
    >>> nw.col("a")._ir.is_scalar()
    False
    >>> nw.col("a").last()._ir.is_scalar()
    True
    """

    __slots__ = ("expr",)
    expr: ExprIR = node()
    """The input for this aggregation.

    >>> import narwhals._plan as nw
    >>> nw.col("a").alias("b").sum()._ir.expr
    col('a').alias('b')

    Note:
        To match the behavior on main ([1], [2]), scalar input is rejected at construction-time.
        However, [`CompliantScalar`] offers an alternative - which can allow a backend to
        more closely resemble `polars`.

    [1]: https://github.com/narwhals-dev/narwhals/blob/061c97f8a01bf9e721835978b039303c5051501c/narwhals/_expression_parsing.py#L495-L498
    [2]: https://github.com/narwhals-dev/narwhals/blob/061c97f8a01bf9e721835978b039303c5051501c/narwhals/_expression_parsing.py#L515-L517
    [`CompliantScalar`]: https://github.com/narwhals-dev/narwhals/blob/7eb0d60159dd485af04d49ec9a1b18d99085482c/narwhals/_plan/compliant/scalar.py
    """

    def is_scalar(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"{self.expr!r}.{self.__expr_ir_dispatch__.name}()"

    if TYPE_CHECKING:
        ...
    else:

        def __init__(self, *, expr: ExprIR, **kwds: Any) -> None:
            # NOTE: Needs to skip type checking to avoid incorrectly synthesized `__replace__` signature
            # https://discuss.python.org/t/dataclass-transform-and-replace/69067
            if expr.is_scalar():
                raise agg_scalar_error(self, expr)
            super().__init__(expr=expr, **kwds)


# fmt: off
class _MomentAggExpr(AggExpr, dtype=map_first(dtm.moment_dtype)): ...
class Count(AggExpr, dtype=dtm.IDX_DTYPE):
    """Non-null count."""
class Len(AggExpr, dtype=dtm.IDX_DTYPE):
    """Null-inclusive count."""
class Max(AggExpr, dtype=same_dtype()): ...
class Mean(_MomentAggExpr): ...
class Median(_MomentAggExpr): ...
class Min(AggExpr, dtype=same_dtype()): ...
class NUnique(AggExpr, dtype=dtm.IDX_DTYPE): ...
class Sum(AggExpr, dtype=map_first(dtm.sum_dtype)): ...
class OrderableAggExpr(AggExpr): ...
class First(OrderableAggExpr, dtype=same_dtype()): ...
class Last(OrderableAggExpr, dtype=same_dtype()): ...
class ArgMin(OrderableAggExpr, dtype=dtm.IDX_DTYPE): ...
class ArgMax(OrderableAggExpr, dtype=dtm.IDX_DTYPE): ...
class Quantile(_MomentAggExpr):
    __slots__ = ("interpolation", "quantile")
    quantile: float
    interpolation: RollingInterpolationMethod
class Std(_MomentAggExpr):
    __slots__ = ("ddof",)
    ddof: int
class Var(AggExpr, dtype=map_first(dtm.var_dtype)):
    __slots__ = ("ddof",)
    ddof: int
# fmt: on
