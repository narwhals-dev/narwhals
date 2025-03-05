from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._compliant.dataframe import CompliantDataFrame
    from narwhals._compliant.dataframe import CompliantLazyFrame
    from narwhals._compliant.expr import CompliantExpr
    from narwhals._compliant.series import CompliantSeries

__all__ = [
    "CompliantDataFrameT",
    "CompliantFrameT",
    "CompliantLazyFrameT",
    "CompliantSeriesT_co",
    "IntoCompliantExpr",
]

CompliantSeriesT_co = TypeVar(
    "CompliantSeriesT_co", bound="CompliantSeries", covariant=True
)
CompliantFrameT = TypeVar(
    "CompliantFrameT", bound="CompliantDataFrame[Any] | CompliantLazyFrame"
)
CompliantDataFrameT = TypeVar("CompliantDataFrameT", bound="CompliantDataFrame[Any]")
CompliantLazyFrameT = TypeVar("CompliantLazyFrameT", bound="CompliantLazyFrame")
IntoCompliantExpr: TypeAlias = (
    "CompliantExpr[CompliantFrameT, CompliantSeriesT_co] | CompliantSeriesT_co"
)
