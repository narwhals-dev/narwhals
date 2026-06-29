from __future__ import annotations

from typing import TYPE_CHECKING, Any

from duckdb import DuckDBPyRelation

from narwhals import _interchange
from narwhals._duckdb import utils
from narwhals._duckdb.dataframe import to_arrow_table
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from types import ModuleType

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Never, Self

    from narwhals._duckdb.utils import DeferredTimeZone
    from narwhals.dtypes import DType


def narwhals_dtype(dtype: Any, time_zone: DeferredTimeZone) -> DType:
    return utils.native_to_narwhals_dtype(dtype, Version.V1, time_zone)


class DuckDBDataFrame:
    _implementation = Implementation.DUCKDB
    _version: Version = Version.V1

    def __init__(self, df: DuckDBPyRelation) -> None:
        self._native_frame = df

    @property
    def native(self) -> DuckDBPyRelation:
        return self._native_frame

    def simple_select(self, *column_names: str) -> DuckDBDataFrame:
        return DuckDBDataFrame(self.native.select(*column_names))

    def get_column(self, name: str) -> DuckDBInterchangeSeries:
        return DuckDBInterchangeSeries(self.native.select(name))

    def to_pandas(self) -> pd.DataFrame:
        return self.native.df()

    def to_arrow(self) -> pa.Table:
        return to_arrow_table(self.native)

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __native_namespace__(self) -> ModuleType:
        return self._implementation.to_native_namespace()

    @property
    def schema(self) -> dict[str, DType]:
        deferred = utils.DeferredTimeZone(self.native)
        it = zip(self.columns, self.native.types, strict=True)
        return {name: narwhals_dtype(dtype, deferred) for name, dtype in it}

    @property
    def columns(self) -> list[str]:
        return self.native.columns

    def collect_schema(self) -> dict[str, DType]:
        return self.schema

    def __getattr__(self, name: str) -> Never:
        raise _interchange.unsupported_error(name)


class DuckDBInterchangeSeries(_interchange.InterchangeSeriesV1[DuckDBPyRelation]):
    _implementation = Implementation.DUCKDB

    def __init__(self, df: DuckDBPyRelation) -> None:
        self._native_series = df

    @property
    def dtype(self) -> DType:
        native = self.native
        return narwhals_dtype(native.types[0], utils.DeferredTimeZone(native))
