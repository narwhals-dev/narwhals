from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Any
import typing as t

from narwhals._plan.common import Function
from narwhals._plan.options import FunctionFlags, FunctionOptions

if t.TYPE_CHECKING:
    from narwhals.typing import ClosedInterval


class BooleanFunction(Function):
    def __repr__(self) -> str:
        tp = type(self)
        if tp is BooleanFunction:
            return tp.__name__
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
            IsIn: "is_in",
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
    """`lower_bound`, `upper_bound` aren't spec'd in the function enum.

    Assuming the `FunctionExpr.input` becomes `s` in the impl

    https://github.com/pola-rs/polars/blob/62257860a43ec44a638e8492ed2cf98a49c05f2e/crates/polars-plan/src/dsl/function_expr/boolean.rs#L225-L237
    """

    __slots__ = ("closed",)

    closed: ClosedInterval

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


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


class IsIn(BooleanFunction):
    """``other` isn't spec'd in the function enum.

    See `IsBetween` comment.
    """

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


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
