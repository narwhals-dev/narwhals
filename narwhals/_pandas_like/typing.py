from __future__ import annotations  # pragma: no cover

from typing import TYPE_CHECKING  # pragma: no cover

if TYPE_CHECKING:
    from typing import TypeAlias

    from narwhals._pandas_like.expr import PandasExpr
    from narwhals._pandas_like.series import PandasSeries

    IntoPandasExpr: TypeAlias = PandasExpr | str | int | float | PandasSeries
