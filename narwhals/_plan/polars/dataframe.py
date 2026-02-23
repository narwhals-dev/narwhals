from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
from typing_extensions import Self

from narwhals._plan._version import into_version
from narwhals._plan.compliant.dataframe import CompliantDataFrame
from narwhals._plan.polars.frame import PolarsFrame
from narwhals._utils import Version, not_implemented

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.polars.lazyframe import PolarsLazyFrame
    from narwhals.schema import Schema


Incomplete: TypeAlias = Any
MAIN = Version.MAIN


class PolarsDataFrame(
    PolarsFrame, CompliantDataFrame[Incomplete, pl.DataFrame, pl.Series]
):
    _native: pl.DataFrame
    _version: Version

    def __len__(self) -> int:
        return self.native.__len__()

    @property
    def columns(self) -> list[str]:
        return self.native.columns

    @property
    def native(self) -> pl.DataFrame:
        return self._native

    @property
    def version(self) -> Version:
        return self._version

    @property
    def schema(self) -> Schema:
        return into_version(self.version).schema.from_polars(self.native.schema)

    @property
    def shape(self) -> tuple[int, int]:
        return self.native.shape

    @classmethod
    def from_polars(cls, frame: pl.DataFrame, /, version: Version = MAIN) -> Self:
        return cls.from_native(frame, version)

    def to_arrow(self) -> pa.Table:
        return self.native.to_arrow()

    def to_pandas(self) -> pd.DataFrame:
        return self.native.to_pandas()

    def to_polars(self) -> pl.DataFrame:
        return self.native

    def to_lazy(self) -> PolarsLazyFrame:
        from narwhals._plan.polars.lazyframe import PolarsLazyFrame

        return PolarsLazyFrame.from_native(self.native.lazy(), self.version)

    __narwhals_namespace__ = not_implemented()
    _evaluate_irs = not_implemented()
    _group_by = not_implemented()  # type: ignore[assignment]
    lazy = not_implemented()
    clone = not_implemented()
    drop = not_implemented()
    drop_nulls = not_implemented()
    explode = not_implemented()
    filter = not_implemented()
    from_dict = not_implemented()
    gather_every = not_implemented()
    get_column = not_implemented()
    iter_columns = not_implemented()
    join = not_implemented()
    join_asof = not_implemented()
    join_cross = not_implemented()
    partition_by = not_implemented()
    pivot = not_implemented()
    rename = not_implemented()
    row = not_implemented()
    sample_n = not_implemented()
    select = not_implemented()
    select_names = not_implemented()
    slice = not_implemented()
    sort = not_implemented()
    to_dict = not_implemented()
    to_series = not_implemented()
    to_struct = not_implemented()
    unique = not_implemented()
    unique_by = not_implemented()
    unnest = not_implemented()
    unpivot = not_implemented()
    with_columns = not_implemented()
    with_row_index = not_implemented()
    with_row_index_by = not_implemented()
    write_csv = not_implemented()
    write_parquet = not_implemented()


PolarsDataFrame()
