from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals import _interchange
from narwhals._ibis.utils import native_to_narwhals_dtype
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from collections.abc import Sequence

    import ibis.expr.types as ir
    import ibis.expr.types.dataframe_interchange as ibis_dfi
    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals.dtypes import DType


class IbisDataFrame(_interchange.InterchangeFrameV1["ir.Table"]):
    _implementation = Implementation.IBIS

    def __init__(self, df: ir.Table | ibis_dfi.IbisDataFrame) -> None:
        self._dfi: ibis_dfi.IbisDataFrame = df.__dataframe__()

    @classmethod
    def from_dfi(cls, df: ibis_dfi.IbisDataFrame) -> Self:
        self = cls.__new__(cls)
        self._dfi = df
        return self

    @property
    def native(self) -> ir.Table:
        return self._dfi._table

    @property
    def _df(self) -> ir.Table:
        return self.native

    def to_pandas(self) -> pd.DataFrame:
        return self.native.to_pandas()

    def to_arrow(self) -> pa.Table:
        return self.native.to_pyarrow()

    def get_column(self, name: str) -> IbisSeries:
        return IbisSeries(self.native.select(name))

    def column_names(self) -> list[str]:
        return list(self.native.columns)

    def select_columns_by_name(self, names: Sequence[str]) -> Self:
        return self.from_dfi(self._dfi.select_columns_by_name(names))

    @property
    def schema(self) -> dict[str, DType]:
        return {
            name: native_to_narwhals_dtype(dtype, self._version)
            for name, dtype in self.native.schema().fields.items()
        }


class IbisSeries(_interchange.InterchangeSeriesV1["ir.Table"]):
    _implementation = Implementation.IBIS

    def __init__(self, df: ir.Table) -> None:
        self._native_series: ir.Table = df

    @property
    def dtype(self) -> DType:
        native = next(iter(self._native_series.schema().values()))
        # NOTE: `Concrete` (base class for `DataType`) has two bases that define `__hash__ = None`,
        # and then decides to implement `__hash__`? Okay ...
        return native_to_narwhals_dtype(native, self._version)  # pyright: ignore[reportArgumentType]
