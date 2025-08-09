from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.typing import NumericLiteral, TemporalLiteral


class IsClose(Protocol):
    """Every member defined is a dependency of `is_close` method."""

    def __and__(self, other: Any) -> Self: ...
    def __or__(self, other: Any) -> Self: ...
    def __invert__(self) -> Self: ...
    def __sub__(self, other: Any) -> Self: ...
    def __mul__(self, other: Any) -> Self: ...
    def __eq__(self, other: Self | Any) -> Self: ...  # type: ignore[override]
    def __gt__(self, other: Any) -> Self: ...
    def __le__(self, other: Any) -> Self: ...
    def abs(self) -> Self: ...
    def is_nan(self) -> Self: ...
    def is_finite(self) -> Self: ...
    def clip(
        self,
        lower_bound: Self | NumericLiteral | TemporalLiteral | None,
        upper_bound: Self | NumericLiteral | TemporalLiteral | None,
    ) -> Self: ...
    def is_close(
        self,
        other: Self | NumericLiteral,
        *,
        abs_tol: float,
        rel_tol: float,
        nans_equal: bool,
    ) -> Self:
        from decimal import Decimal

        other_abs: Self | NumericLiteral
        other_is_nan: Self | bool
        other_is_inf: Self | bool
        other_is_not_inf: Self | bool

        if isinstance(other, (float, int, Decimal)):
            from math import isinf, isnan

            other_abs = other.__abs__()
            other_is_nan = isnan(other)
            other_is_inf = isinf(other)

            # Define the other_is_not_inf variable to prevent triggering the following warning:
            # > DeprecationWarning: Bitwise inversion '~' on bool is deprecated and will be
            # >     removed in Python 3.16.
            other_is_not_inf = not other_is_inf

        else:
            other_abs, other_is_nan = other.abs(), other.is_nan()
            other_is_not_inf = other.is_finite() | other_is_nan
            other_is_inf = ~other_is_not_inf

        rel_threshold = self.abs().clip(lower_bound=other_abs, upper_bound=None) * rel_tol
        tolerance = rel_threshold.clip(lower_bound=abs_tol, upper_bound=None)

        self_is_nan = self.is_nan()
        self_is_not_inf = self.is_finite() | self_is_nan

        # Values are close if abs_diff <= tolerance, and both finite
        is_close = (
            ((self - other).abs() <= tolerance) & self_is_not_inf & other_is_not_inf
        )

        # Handle infinity cases: infinities are close/equal if they have the same sign
        self_sign, other_sign = self > 0, other > 0
        is_same_inf = (~self_is_not_inf) & other_is_inf & (self_sign == other_sign)

        # Handle nan cases:
        #   * If any value is NaN, then False (via `& ~either_nan`)
        #   * However, if `nans_equals = True` and if _both_ values are NaN, then True
        either_nan = self_is_nan | other_is_nan
        result = (is_close | is_same_inf) & ~either_nan

        if nans_equal:
            both_nan = self_is_nan & other_is_nan
            result = result | both_nan

        return result
