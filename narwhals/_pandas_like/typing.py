from __future__ import annotations  # pragma: no cover

from typing import TYPE_CHECKING  # pragma: no cover

if TYPE_CHECKING:
    from typing import Any, TypeVar

    import pandas as pd
    from typing_extensions import TypeAlias

    from narwhals._pandas_like.expr import PandasLikeExpr
    from narwhals._pandas_like.series import PandasLikeSeries

    IntoPandasLikeExpr: TypeAlias = "PandasLikeExpr | PandasLikeSeries"
    NDFrameT = TypeVar("NDFrameT", "pd.DataFrame", "pd.Series[Any]")
