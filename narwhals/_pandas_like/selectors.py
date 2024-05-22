from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._pandas_like.expr import PandasExpr

if TYPE_CHECKING:
    from narwhals._pandas_like.dataframe import PandasDataFrame
    from narwhals._pandas_like.series import PandasSeries
    from narwhals.dtypes import DType


class PandasSelector:
    def __init__(self, implementation: str) -> None:
        self._implementation = implementation

    def by_dtype(self, dtypes: list[DType]) -> PandasExpr:
        def func(df: PandasDataFrame) -> list[PandasSeries]:
            return [df[col] for col in df.columns if df.schema[col] in dtypes]

        return PandasExpr(
            func,
            depth=0,
            function_name="type_selector",
            root_names=None,
            output_names=None,
            implementation=self._implementation,
        )
