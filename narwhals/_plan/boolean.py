from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Any
import typing as t

from narwhals._plan.common import Function, HorizontalFunction
from narwhals._plan.options import FConfig, FunctionOptions
from narwhals._typing_compat import TypeVar

if t.TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.common import ExprIR
    from narwhals._plan.dummy import Series
    from narwhals._plan.expr import FunctionExpr, Literal  # noqa: F401
    from narwhals._plan.typing import NativeSeriesT, Seq  # noqa: F401
    from narwhals.typing import ClosedInterval

OtherT = TypeVar("OtherT")
ExprT = TypeVar("ExprT", bound="ExprIR", default="ExprIR")


class BooleanFunction(Function): ...


class All(BooleanFunction, options=FunctionOptions.aggregation): ...


class AllHorizontal(HorizontalFunction, BooleanFunction): ...


class Any(BooleanFunction, options=FunctionOptions.aggregation): ...


class AnyHorizontal(HorizontalFunction, BooleanFunction): ...


class IsBetween(BooleanFunction, options=FunctionOptions.elementwise):
    """N-ary (expr, lower_bound, upper_bound)."""

    __slots__ = ("closed",)
    closed: ClosedInterval

    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR, ExprIR]:
        expr, lower_bound, upper_bound = node.input
        return expr, lower_bound, upper_bound


class IsDuplicated(BooleanFunction, options=FunctionOptions.length_preserving): ...


class IsFinite(BooleanFunction, options=FunctionOptions.elementwise): ...


class IsFirstDistinct(BooleanFunction, options=FunctionOptions.length_preserving): ...


class IsIn(BooleanFunction, t.Generic[OtherT], options=FunctionOptions.elementwise):
    __slots__ = ("other",)
    other: OtherT

    def __repr__(self) -> str:
        return "is_in"


class IsInSeq(IsIn["Seq[t.Any]"]):
    @classmethod
    def from_iterable(cls, other: t.Iterable[t.Any], /) -> IsInSeq:
        if not isinstance(other, (str, bytes)):
            return IsInSeq(other=tuple(other))
        msg = f"`is_in` doesn't accept `str | bytes` as iterables, got: {type(other).__name__}"
        raise TypeError(msg)


# NOTE: Shouldn't be allowed for lazy backends (maybe besides `polars`)
class IsInSeries(IsIn["Literal[Series[NativeSeriesT]]"]):
    @classmethod
    def from_series(cls, other: Series[NativeSeriesT], /) -> IsInSeries[NativeSeriesT]:
        from narwhals._plan.literal import SeriesLiteral

        return IsInSeries(other=SeriesLiteral(value=other).to_literal())


# NOTE: Placeholder for allowing `Expr` iff it passes `.meta.is_column()`
class IsInExpr(IsIn[ExprT], t.Generic[ExprT]):
    def __init__(self, *, other: ExprT) -> None:
        msg = (
            "`is_in` doesn't accept expressions as an argument, as opposed to Polars. "
            "You should provide an iterable instead."
        )
        raise NotImplementedError(msg)


class IsLastDistinct(BooleanFunction, options=FunctionOptions.length_preserving): ...


class IsNan(BooleanFunction, options=FunctionOptions.elementwise): ...


class IsNull(BooleanFunction, options=FunctionOptions.elementwise): ...


class IsUnique(BooleanFunction, options=FunctionOptions.length_preserving): ...


class Not(
    BooleanFunction, options=FunctionOptions.elementwise, config=FConfig.renamed("not_")
): ...
