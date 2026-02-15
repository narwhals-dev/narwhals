from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Any
import typing as t
from typing import TYPE_CHECKING

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._function import Function, HorizontalFunction
from narwhals._plan.options import FEOptions, FunctionOptions
from narwhals._plan.typing import NativeSeriesT

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan._expr_ir import ExprIR
    from narwhals._plan.expressions.expr import FunctionExpr as FExpr, Literal
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.series import Series
    from narwhals._plan.typing import Seq
    from narwhals.dtypes import DType
    from narwhals.typing import ClosedInterval


# fmt: off
class BooleanFunction(Function, options=FunctionOptions.elementwise):
    def _resolve_dtype(self, schema: FrozenSchema, node: FExpr[Function]) -> DType:
        return dtm.BOOLEAN_DTYPE
class _HorizontalBoolean(HorizontalFunction, BooleanFunction):
    __slots__ = ("ignore_nulls",)
    ignore_nulls: bool
    _resolve_dtype = BooleanFunction._resolve_dtype
class All(BooleanFunction, options=FunctionOptions.aggregation): ...
class Any(BooleanFunction, options=FunctionOptions.aggregation): ...
class AllHorizontal(_HorizontalBoolean):...
class AnyHorizontal(_HorizontalBoolean):...
class IsDuplicated(BooleanFunction, options=FunctionOptions.length_preserving): ...
class IsFinite(BooleanFunction): ...
class IsFirstDistinct(BooleanFunction, options=FunctionOptions.length_preserving): ...
class IsLastDistinct(BooleanFunction, options=FunctionOptions.length_preserving): ...
class IsNan(BooleanFunction): ...
class IsNull(BooleanFunction): ...
class IsNotNan(BooleanFunction): ...
class IsNotNull(BooleanFunction): ...
class IsUnique(BooleanFunction, options=FunctionOptions.length_preserving): ...
class Not(BooleanFunction, config=FEOptions.renamed("not_")):
    def _resolve_dtype(self, schema: FrozenSchema, node: FExpr[Function]) -> DType:
        # https://github.com/pola-rs/polars/pull/14049
        dtype = node.input[0]._resolve_dtype(schema)
        return dtype if dtype.is_integer() else dtm.BOOLEAN_DTYPE
# fmt: on
class IsBetween(BooleanFunction):
    """N-ary (expr, lower_bound, upper_bound)."""

    __slots__ = ("closed",)
    closed: ClosedInterval

    def unwrap_input(self, node: FExpr[Self], /) -> tuple[ExprIR, ExprIR, ExprIR]:
        expr, lower_bound, upper_bound = node.input
        return expr, lower_bound, upper_bound


class IsInSeq(BooleanFunction):
    __slots__ = ("other",)
    other: Seq[t.Any]

    def __repr__(self) -> str:
        return "is_in"

    @classmethod
    def from_iterable(cls, other: t.Iterable[t.Any], /) -> IsInSeq:
        if not isinstance(other, (str, bytes)):
            return IsInSeq(other=tuple(other))
        msg = f"`is_in` doesn't accept `str | bytes` as iterables, got: {type(other).__name__}"
        raise TypeError(msg)


class IsInSeries(BooleanFunction, t.Generic[NativeSeriesT]):
    __slots__ = ("other",)
    other: Literal[Series[NativeSeriesT]]

    def __repr__(self) -> str:
        return "is_in"

    @classmethod
    def from_series(cls, other: Series[NativeSeriesT], /) -> IsInSeries[NativeSeriesT]:
        from narwhals._plan.expressions.literal import SeriesLiteral

        return IsInSeries(other=SeriesLiteral(value=other).to_literal())


class IsInExpr(BooleanFunction):
    """N-ary (expr, other).

    Note:
        If we get to a stage where `narwhals` has wide support for `list`, and
        accepts them in `lit(...)` - *consider* [restricting to non-equal types].

    [restricting to non-equal types]: https://github.com/pola-rs/polars/pull/22178
    """

    def unwrap_input(self, node: FExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, other = node.input
        return expr, other

    def __repr__(self) -> str:
        return "is_in"


def all_horizontal(*exprs: ExprIR, ignore_nulls: bool = False) -> FExpr[AllHorizontal]:
    return AllHorizontal(ignore_nulls=ignore_nulls).to_function_expr(*exprs)
