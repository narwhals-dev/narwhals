from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals import dtypes
from narwhals._pandas_like.expr import PandasExpr

if TYPE_CHECKING:
    from narwhals._pandas_like.dataframe import PandasDataFrame
    from narwhals._pandas_like.series import PandasSeries
    from narwhals.dtypes import DType


class PandasSelector:
    def __init__(self, implementation: str) -> None:
        self._implementation = implementation

    def by_dtype(self, dtypes: list[DType | type[DType]]) -> PandasExpr:
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

    def numeric(self) -> PandasExpr:
        return self.by_dtype(
            [
                dtypes.Int64,
                dtypes.Int32,
                dtypes.Int16,
                dtypes.Int8,
                dtypes.UInt64,
                dtypes.UInt32,
                dtypes.UInt16,
                dtypes.UInt8,
                dtypes.Float64,
                dtypes.Float32,
            ]
        )

    def boolean(self) -> PandasExpr:
        return self.by_dtype([dtypes.Boolean])

    def string(self) -> PandasExpr:
        return self.by_dtype([dtypes.String])

    def categorical(self) -> PandasExpr:
        return self.by_dtype([dtypes.Categorical])
