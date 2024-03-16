from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from narwhals.pandas_like.dataframe import PandasDataFrame
    from narwhals.pandas_like.series import PandasSeries


def translate_frame(
    df: Any,
    implementation: str,
) -> PandasDataFrame:
    from narwhals.pandas_like.dataframe import PandasDataFrame

    return PandasDataFrame(
        df,
        implementation=implementation,
    )


def translate_series(
    series: Any,
    implementation: str,
) -> PandasSeries:
    from narwhals.pandas_like.series import PandasSeries

    return PandasSeries(series, implementation=implementation)
