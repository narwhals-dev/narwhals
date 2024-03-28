from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeAlias

    from narwhals.pandas_like.expr import PandasExpr
    from narwhals.pandas_like.series import PandasSeries

    IntoPandasExpr: TypeAlias = PandasExpr | str | int | float | PandasSeries
