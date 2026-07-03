from __future__ import annotations

from typing import TYPE_CHECKING

from duckdb import DuckDBPyRelation

from narwhals import _interchange
from narwhals._duckdb.dataframe import to_arrow_table
from narwhals._utils import Implementation

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


class DuckDBDataFrame(_interchange.LazyFrame[DuckDBPyRelation]):
    _implementation = Implementation.DUCKDB

    def to_pandas(self) -> pd.DataFrame:
        return self._compliant.native.df()

    def to_arrow(self) -> pa.Table:
        return to_arrow_table(self._compliant.native)
