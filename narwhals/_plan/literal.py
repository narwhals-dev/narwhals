from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from narwhals._plan.common import Immutable

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._plan.dummy import DummySeries
    from narwhals._plan.expr import Literal
    from narwhals.dtypes import DType
    from narwhals.typing import NonNestedLiteral

from narwhals._typing_compat import TypeVar

T = TypeVar("T", default=Any)
NonNestedLiteralT = TypeVar(
    "NonNestedLiteralT", bound="NonNestedLiteral", default="NonNestedLiteral"
)


class LiteralValue(Immutable, Generic[T]):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/plans/lit.rs#L67-L73."""

    @property
    def dtype(self) -> DType:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "literal"

    @property
    def is_scalar(self) -> bool:
        return False

    def to_literal(self) -> Literal:
        from narwhals._plan.expr import Literal

        return Literal(value=self)

    def unwrap(self) -> T:
        raise NotImplementedError


class ScalarLiteral(LiteralValue[NonNestedLiteralT]):
    __slots__ = ("dtype", "value")

    value: NonNestedLiteralT
    dtype: DType

    @property
    def is_scalar(self) -> bool:
        return True

    def __repr__(self) -> str:
        if self.value is not None:
            return f"{type(self.value).__name__}: {self.value!s}"
        return "null"

    def unwrap(self) -> NonNestedLiteralT:
        return self.value


class SeriesLiteral(LiteralValue["DummySeries"]):
    """We already need this.

    https://github.com/narwhals-dev/narwhals/blob/e51eba891719a5eb1f7ce91c02a477af39c0baee/narwhals/_expression_parsing.py#L96-L97
    """

    __slots__ = ("value",)

    value: DummySeries

    @property
    def dtype(self) -> DType:
        return self.value.dtype

    @property
    def name(self) -> str:
        return self.value.name

    def __repr__(self) -> str:
        return "Series"

    def unwrap(self) -> DummySeries:
        return self.value


class RangeLiteral(LiteralValue):
    """Don't need yet, but might push forward the discussions.

    - https://github.com/narwhals-dev/narwhals/issues/2463#issuecomment-2844654064
    - https://github.com/narwhals-dev/narwhals/issues/2307#issuecomment-2832422364.
    """

    __slots__ = ("dtype", "high", "low")

    low: int
    high: int
    dtype: DType


def is_scalar_literal(obj: Any) -> TypeIs[ScalarLiteral]:
    return isinstance(obj, ScalarLiteral)


def is_series_literal(obj: Any) -> TypeIs[SeriesLiteral]:
    return isinstance(obj, SeriesLiteral)
