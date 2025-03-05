from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

if TYPE_CHECKING:
    from narwhals._compliant.dataframe import CompliantDataFrame
    from narwhals._compliant.dataframe import CompliantLazyFrame
    from narwhals._compliant.series import CompliantSeries

CompliantSeriesT_co = TypeVar(
    "CompliantSeriesT_co", bound="CompliantSeries", covariant=True
)
CompliantFrameT = TypeVar(
    "CompliantFrameT", bound="CompliantDataFrame[Any] | CompliantLazyFrame"
)
