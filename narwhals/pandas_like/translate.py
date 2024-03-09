from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from narwhals.pandas_like.dataframe import PdxDataFrame
    from narwhals.pandas_like.namespace import Namespace
    from narwhals.pandas_like.series import PdxSeries


def translate_frame(
    df: Any,
    implementation: str,
    *,
    is_eager: bool,
    is_lazy: bool,
) -> tuple[PdxDataFrame, Namespace]:
    from narwhals.pandas_like.dataframe import PdxDataFrame
    from narwhals.pandas_like.namespace import Namespace

    df = PdxDataFrame(
        df,
        implementation=implementation,
        is_eager=is_eager,
        is_lazy=is_lazy,
    )
    return df, Namespace(implementation=implementation)


def translate_series(
    series: Any,
    implementation: str,
) -> tuple[PdxSeries, Namespace]:
    from narwhals.pandas_like.namespace import Namespace
    from narwhals.pandas_like.series import PdxSeries

    return PdxSeries(series, implementation=implementation), Namespace(
        implementation=implementation
    )
