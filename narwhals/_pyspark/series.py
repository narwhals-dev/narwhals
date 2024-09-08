from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._pyspark.utils import translate_sql_api_dtype
from narwhals._pyspark.utils import validate_column_comparand
from narwhals.dependencies import get_pyspark_sql
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from typing_extensions import Self

    from narwhals.dtypes import DType


def _check_one_col_df(dataframe: DataFrame, col_name: str) -> None:
    columns = dataframe.columns
    if len(columns) != 1:
        msg = "Internal DataFrame in PySparkSeries must have exactly one column"
        raise ValueError(msg)
    if columns[0] != col_name:
        msg = f"Internal DataFrame column name must be {col_name}"
        raise ValueError(msg)


class PySparkSeries:
    def __init__(self, native_series: DataFrame, *, name: str) -> None:
        import pyspark.sql.functions as F  # noqa: N812

        _check_one_col_df(native_series, name)
        self._name = name
        self._native_column = F.col(name)
        self._native_series = native_series
        self._implementation = Implementation.PYSPARK

    def __native_namespace__(self) -> Any:
        return get_pyspark_sql()

    def __narwhals_series__(self) -> Self:
        return self

    def _from_native_series(self, series: DataFrame) -> Self:
        return self.__class__(series, name=self._name)

    def __len__(self) -> int:
        return self._native_series.count()

    def __eq__(self, other: object) -> Self:
        other = validate_column_comparand(other)
        current_name = self._name
        new_column = (self._native_column == other).cast("boolean").alias(current_name)
        return self._from_native_series(self._native_series.select(new_column))

    def __ne__(self, other: object) -> Self:
        other = validate_column_comparand(other)
        current_name = self._name
        new_column = (self._native_column != other).cast("boolean").alias(current_name)
        return self._from_native_series(self._native_series.select(new_column))

    def __gt__(self, other: object) -> Self:
        other = validate_column_comparand(other)
        current_name = self._name
        new_column = (self._native_column > other).cast("boolean").alias(current_name)
        print(self._native_series)
        print(new_column)
        return self._from_native_series(self._native_series.select(new_column))

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)  # type: ignore[no-any-return]

    @property
    def dtype(self) -> DType:
        schema_ = self._native_series.schema
        return translate_sql_api_dtype(schema_[0].dataType)

    def alias(self, name: str) -> Self:
        return self.__class__(self._native_series.withColumn(name, self.name), name=name)
