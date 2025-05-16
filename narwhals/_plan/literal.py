from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprIR

if TYPE_CHECKING:
    from narwhals._plan.common import DummySeries
    from narwhals.dtypes import DType
    from narwhals.typing import PythonLiteral


class LiteralValue(ExprIR):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/plans/lit.rs#L67-L73."""


class ScalarLiteral(LiteralValue):
    __slots__ = ("value",)

    value: PythonLiteral

    @property
    def is_scalar(self) -> bool:
        return True

    def __repr__(self) -> str:
        if self.value is not None:
            return f"{type(self.value).__name__}: {self.value}"
        return "null"


class SeriesLiteral(LiteralValue):
    """We already need this.

    https://github.com/narwhals-dev/narwhals/blob/e51eba891719a5eb1f7ce91c02a477af39c0baee/narwhals/_expression_parsing.py#L96-L97
    """

    __slots__ = ("value",)

    value: DummySeries

    def __repr__(self) -> str:
        return "Series"


class RangeLiteral(LiteralValue):
    """Don't need yet, but might push forward the discussions.

    - https://github.com/narwhals-dev/narwhals/issues/2463#issuecomment-2844654064
    - https://github.com/narwhals-dev/narwhals/issues/2307#issuecomment-2832422364.
    """

    __slots__ = ("dtype", "high", "low")

    low: int
    high: int
    dtype: DType
