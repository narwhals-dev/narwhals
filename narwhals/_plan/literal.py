from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from narwhals._plan.common import Immutable
from narwhals._plan.typing import LiteralT, NativeSeriesT, NonNestedLiteralT

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._plan.dummy import DummySeries
    from narwhals._plan.expr import Literal
    from narwhals.dtypes import DType


class LiteralValue(Immutable, Generic[LiteralT]):
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

    def to_literal(self) -> Literal[LiteralT]:
        from narwhals._plan.expr import Literal

        return Literal(value=self)

    def unwrap(self) -> LiteralT:
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


class SeriesLiteral(LiteralValue["DummySeries[NativeSeriesT]"]):
    """We already need this.

    https://github.com/narwhals-dev/narwhals/blob/e51eba891719a5eb1f7ce91c02a477af39c0baee/narwhals/_expression_parsing.py#L96-L97
    """

    __slots__ = ("value",)
    value: DummySeries[NativeSeriesT]

    @property
    def dtype(self) -> DType:
        return self.value.dtype

    @property
    def name(self) -> str:
        return self.value.name

    def __repr__(self) -> str:
        return "Series"

    def unwrap(self) -> DummySeries[NativeSeriesT]:
        return self.value


def _is_scalar(
    obj: ScalarLiteral[NonNestedLiteralT] | Any,
) -> TypeIs[ScalarLiteral[NonNestedLiteralT]]:
    return isinstance(obj, ScalarLiteral)


def _is_series(
    obj: SeriesLiteral[NativeSeriesT] | Any,
) -> TypeIs[SeriesLiteral[NativeSeriesT]]:
    return isinstance(obj, SeriesLiteral)


def is_literal(obj: Literal[LiteralT] | Any) -> TypeIs[Literal[LiteralT]]:
    from narwhals._plan.expr import Literal

    return isinstance(obj, Literal)


def is_literal_scalar(
    obj: Literal[NonNestedLiteralT] | Any,
) -> TypeIs[Literal[NonNestedLiteralT]]:
    return is_literal(obj) and _is_scalar(obj.value)


def is_literal_series(
    obj: Literal[DummySeries[NativeSeriesT]] | Any,
) -> TypeIs[Literal[DummySeries[NativeSeriesT]]]:
    return is_literal(obj) and _is_series(obj.value)
