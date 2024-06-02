from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals import dtypes
from narwhals._pandas_like.expr import PandasExpr

if TYPE_CHECKING:
    from narwhals._pandas_like.dataframe import PandasDataFrame
    from narwhals._pandas_like.series import PandasSeries
    from narwhals.dtypes import DType


def by_dtype(dtypes: list[DType | type[DType]], implementation: str) -> PandasExpr:
    def func(df: PandasDataFrame) -> list[PandasSeries]:
        return [df[col] for col in df.columns if df.schema[col] in dtypes]

    return PandasSelector(
        func,
        depth=0,
        function_name="type_selector",
        root_names=None,
        output_names=None,
        implementation=implementation,
    )


def numeric(implementation):
    return by_dtype([dtypes.Int64, dtypes.Float64], implementation=implementation)


class PandasSelector(PandasExpr):
    def __sub__(self, other):
        if isinstance(other, PandasSelector):

            def func(plx):
                breakpoint()
                return self._call(plx) - other._call(plx)

            return PandasSelector(func)
