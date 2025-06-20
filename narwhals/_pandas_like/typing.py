from __future__ import annotations  # pragma: no cover

from typing import TYPE_CHECKING  # pragma: no cover

from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd
    from typing_extensions import TypeAlias

    from narwhals._namespace import _NativePandasLikeDataFrame, _NativePandasLikeSeries
    from narwhals._pandas_like.expr import PandasLikeExpr
    from narwhals._pandas_like.series import PandasLikeSeries

    IntoPandasLikeExpr: TypeAlias = "PandasLikeExpr | PandasLikeSeries"
    NDFrameT = TypeVar("NDFrameT", "pd.DataFrame", "pd.Series[Any]")

NativeSeriesT = TypeVar(
    "NativeSeriesT", bound="_NativePandasLikeSeries", default="pd.Series[Any]"
)
NativeDataFrameT = TypeVar(
    "NativeDataFrameT", bound="_NativePandasLikeDataFrame", default="pd.DataFrame"
)
