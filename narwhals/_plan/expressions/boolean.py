from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Any
import typing as t

from narwhals._plan._function import Function, HorizontalFunction
from narwhals._plan.options import FEOptions, FunctionOptions
from narwhals._typing_compat import TypeVar

if t.TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.common import ExprIR
    from narwhals._plan.expressions.expr import FunctionExpr, Literal  # noqa: F401
    from narwhals._plan.series import Series
    from narwhals._plan.typing import NativeSeriesT, Seq  # noqa: F401
    from narwhals.typing import ClosedInterval

OtherT = TypeVar("OtherT")
ExprT = TypeVar("ExprT", bound="ExprIR", default="ExprIR")


# fmt: off
class BooleanFunction(Function, options=FunctionOptions.elementwise): ...
class All(BooleanFunction, options=FunctionOptions.aggregation): ...
class AllHorizontal(HorizontalFunction, BooleanFunction): ...
class Any(BooleanFunction, options=FunctionOptions.aggregation): ...
class AnyHorizontal(HorizontalFunction, BooleanFunction): ...
class IsDuplicated(BooleanFunction, options=FunctionOptions.length_preserving): ...
class IsFinite(BooleanFunction): ...
class IsFirstDistinct(BooleanFunction, options=FunctionOptions.length_preserving): ...
class IsLastDistinct(BooleanFunction, options=FunctionOptions.length_preserving): ...
class IsNan(BooleanFunction): ...
class IsNull(BooleanFunction): ...
class IsUnique(BooleanFunction, options=FunctionOptions.length_preserving): ...
class Not(BooleanFunction, config=FEOptions.renamed("not_")): ...
# fmt: on
class IsBetween(BooleanFunction):
    """N-ary (expr, lower_bound, upper_bound)."""

    __slots__ = ("closed",)
    closed: ClosedInterval

    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR, ExprIR]:
        expr, lower_bound, upper_bound = node.input
        return expr, lower_bound, upper_bound


class IsIn(BooleanFunction, t.Generic[OtherT]):
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
        from narwhals._plan.expressions.literal import SeriesLiteral

        return IsInSeries(other=SeriesLiteral(value=other).to_literal())


# NOTE: Placeholder for allowing `Expr` iff it passes `.meta.is_column()`
class IsInExpr(IsIn[ExprT], t.Generic[ExprT]):
    def __init__(self, *, other: ExprT) -> None:
        msg = (
            "`is_in` doesn't accept expressions as an argument, as opposed to Polars. "
            "You should provide an iterable instead."
        )
        raise NotImplementedError(msg)
