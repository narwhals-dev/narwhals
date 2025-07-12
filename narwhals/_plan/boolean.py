from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Any
import typing as t

from narwhals._plan.common import Function
from narwhals._plan.options import FunctionFlags, FunctionOptions
from narwhals._typing_compat import TypeVar

if t.TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.common import ExprIR
    from narwhals._plan.dummy import DummySeries
    from narwhals._plan.expr import FunctionExpr, Literal  # noqa: F401
    from narwhals._plan.typing import NativeSeriesT, Seq  # noqa: F401
    from narwhals.typing import ClosedInterval

OtherT = TypeVar("OtherT")
ExprT = TypeVar("ExprT", bound="ExprIR", default="ExprIR")


class BooleanFunction(Function):
    def __repr__(self) -> str:
        tp = type(self)
        if tp in {BooleanFunction, IsIn}:
            return tp.__name__
        if isinstance(self, IsIn):
            return "is_in"
        m: dict[type[BooleanFunction], str] = {
            All: "all",
            Any: "any",
            AllHorizontal: "all_horizontal",
            AnyHorizontal: "any_horizontal",
            IsBetween: "is_between",
            IsDuplicated: "is_duplicated",
            IsFinite: "is_finite",
            IsNan: "is_nan",
            IsNull: "is_null",
            IsFirstDistinct: "is_first_distinct",
            IsLastDistinct: "is_last_distinct",
            IsUnique: "is_unique",
            Not: "not",
        }
        return m[tp]


class All(BooleanFunction):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.aggregation()


class AllHorizontal(BooleanFunction):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise().with_flags(
            FunctionFlags.INPUT_WILDCARD_EXPANSION
        )


class Any(BooleanFunction):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.aggregation()


class AnyHorizontal(BooleanFunction):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise().with_flags(
            FunctionFlags.INPUT_WILDCARD_EXPANSION
        )


class IsBetween(BooleanFunction):
    """N-ary (expr, lower_bound, upper_bound)."""

    __slots__ = ("closed",)
    closed: ClosedInterval

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR, ExprIR]:
        expr, lower_bound, upper_bound = node.input
        return expr, lower_bound, upper_bound


class IsDuplicated(BooleanFunction):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.length_preserving()


class IsFinite(BooleanFunction):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


class IsFirstDistinct(BooleanFunction):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.length_preserving()


class IsIn(BooleanFunction, t.Generic[OtherT]):
    __slots__ = ("other",)
    other: OtherT

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


class IsInSeq(IsIn["Seq[t.Any]"]):
    @classmethod
    def from_iterable(cls, other: t.Iterable[t.Any], /) -> IsInSeq:
        if not isinstance(other, (str, bytes)):
            return IsInSeq(other=tuple(other))
        msg = f"`is_in` doesn't accept `str | bytes` as iterables, got: {type(other).__name__}"
        raise TypeError(msg)


# NOTE: Shouldn't be allowed for lazy backends (maybe besides `polars`)
class IsInSeries(IsIn["Literal[DummySeries[NativeSeriesT]]"]):
    @classmethod
    def from_series(
        cls, other: DummySeries[NativeSeriesT], /
    ) -> IsInSeries[NativeSeriesT]:
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


class IsLastDistinct(BooleanFunction):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.length_preserving()


class IsNan(BooleanFunction):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


class IsNull(BooleanFunction):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


class IsUnique(BooleanFunction):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.length_preserving()


class Not(BooleanFunction):
    """`__invert__`."""

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()
