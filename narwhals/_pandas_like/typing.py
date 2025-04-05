from __future__ import annotations  # pragma: no cover

from typing import TYPE_CHECKING  # pragma: no cover
from typing import Union  # pragma: no cover

if TYPE_CHECKING:
    import sys
    from typing import Any

    if sys.version_info >= (3, 13):
        from typing import TypeVar
    else:
        from typing_extensions import TypeVar

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    import cudf
    import modin.pandas as mpd
    import pandas as pd

    from narwhals._pandas_like.expr import PandasLikeExpr
    from narwhals._pandas_like.series import PandasLikeSeries

    IntoPandasLikeExpr: TypeAlias = Union[PandasLikeExpr, PandasLikeSeries]

    DataFrameT = TypeVar(
        "DataFrameT", pd.DataFrame, mpd.DataFrame, cudf.DataFrame, default=pd.DataFrame
    )
    SeriesT = TypeVar(
        "SeriesT", pd.Series[Any], mpd.Series, cudf.Series[Any], default=pd.Series[Any]
    )
    NDFrameT = TypeVar(
        "NDFrameT",
        pd.DataFrame,
        mpd.DataFrame,
        cudf.DataFrame,
        pd.Series[Any],
        mpd.Series,
        cudf.Series[Any],
        default=pd.DataFrame,
    )
