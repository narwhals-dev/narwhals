from __future__ import annotations

from collections.abc import Collection, Iterable, Sized
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import duckdb
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from sqlframe.base.dataframe import BaseDataFrame as _BaseDataFrame
    from typing_extensions import Self, TypeAlias

    SQLFrameDataFrame = _BaseDataFrame[Any, Any, Any, Any, Any]

__all__ = [
    "NativeAny",
    "NativeArrow",
    "NativeCuDF",
    "NativeDask",
    "NativeDataFrame",
    "NativeDuckDB",
    "NativeFrame",
    "NativeIbis",
    "NativeKnown",
    "NativeLazyFrame",
    "NativeModin",
    "NativePandas",
    "NativePandasLike",
    "NativePandasLikeDataFrame",
    "NativePandasLikeSeries",
    "NativePolars",
    "NativePySpark",
    "NativePySparkConnect",
    "NativeSQLFrame",
    "NativeSeries",
    "NativeSparkLike",
    "NativeUnknown",
]


Incomplete: TypeAlias = Any


# All dataframes supported by Narwhals have a
# `columns` property. Their similarities don't extend
# _that_ much further unfortunately...
class NativeFrame(Protocol):
    @property
    def columns(self) -> Any: ...
    def join(self, *args: Any, **kwargs: Any) -> Any: ...


class NativeDataFrame(Sized, NativeFrame, Protocol): ...


class NativeLazyFrame(NativeFrame, Protocol):
    def explain(self, *args: Any, **kwargs: Any) -> Any: ...


class NativeSeries(Sized, Iterable[Any], Protocol):
    def filter(self, *args: Any, **kwargs: Any) -> Any: ...


class _BasePandasLike(Sized, Protocol):
    index: Any
    """`mypy` doesn't like the asymmetric `property` setter in `pandas`."""

    def __getitem__(self, key: Any, /) -> Any: ...
    def __mul__(self, other: float | Collection[float] | Self, /) -> Self: ...
    def __floordiv__(self, other: float | Collection[float] | Self, /) -> Self: ...
    @property
    def loc(self) -> Any: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def set_axis(self, labels: Any, *, axis: Any = ..., copy: bool = ...) -> Self: ...
    def copy(self, deep: bool = ...) -> Self: ...  # noqa: FBT001
    def rename(self, *args: Any, **kwds: Any) -> Self | Incomplete:
        """`mypy` & `pyright` disagree on overloads.

        `Incomplete` used to fix [more important issue](https://github.com/narwhals-dev/narwhals/pull/3016#discussion_r2296139744).
        """


class _BasePandasLikeFrame(NativeDataFrame, _BasePandasLike, Protocol): ...


class _BasePandasLikeSeries(NativeSeries, _BasePandasLike, Protocol):
    def where(self, cond: Any, other: Any = ..., /) -> Self | Incomplete: ...


class NativeDask(NativeLazyFrame, Protocol):
    _partition_type: type[pd.DataFrame]


class _CuDFDataFrame(_BasePandasLikeFrame, Protocol):
    def to_pylibcudf(self, *args: Any, **kwds: Any) -> Any: ...


class _CuDFSeries(_BasePandasLikeSeries, Protocol):
    def to_pylibcudf(self, *args: Any, **kwds: Any) -> Any: ...


class NativeIbis(Protocol):
    def sql(self, *args: Any, **kwds: Any) -> Any: ...
    def __pyarrow_result__(self, *args: Any, **kwds: Any) -> Any: ...
    def __pandas_result__(self, *args: Any, **kwds: Any) -> Any: ...
    def __polars_result__(self, *args: Any, **kwds: Any) -> Any: ...


class _ModinDataFrame(_BasePandasLikeFrame, Protocol):
    _pandas_class: type[pd.DataFrame]


class _ModinSeries(_BasePandasLikeSeries, Protocol):
    _pandas_class: type[pd.Series[Any]]


# NOTE: Using `pyspark.sql.DataFrame` creates false positives in overloads when not installed
class _PySparkDataFrame(NativeLazyFrame, Protocol):
    # Arbitrary method that `sqlframe` doesn't have and unlikely to appear anywhere else
    # https://github.com/apache/spark/blob/8530444e25b83971da4314c608aa7d763adeceb3/python/pyspark/sql/dataframe.py#L4875
    def dropDuplicatesWithinWatermark(self, *arg: Any, **kwargs: Any) -> Any: ...  # noqa: N802


NativePolars: TypeAlias = "pl.DataFrame | pl.LazyFrame | pl.Series"
NativeArrow: TypeAlias = "pa.Table | pa.ChunkedArray[Any]"
NativeDuckDB: TypeAlias = "duckdb.DuckDBPyRelation"
NativePandas: TypeAlias = "pd.DataFrame | pd.Series[Any]"
NativeModin: TypeAlias = "_ModinDataFrame | _ModinSeries"
NativeCuDF: TypeAlias = "_CuDFDataFrame | _CuDFSeries"
NativePandasLikeSeries: TypeAlias = "pd.Series[Any] | _CuDFSeries | _ModinSeries"
NativePandasLikeDataFrame: TypeAlias = "pd.DataFrame | _CuDFDataFrame | _ModinDataFrame"
NativePandasLike: TypeAlias = "NativePandasLikeDataFrame | NativePandasLikeSeries"
NativeSQLFrame: TypeAlias = "_BaseDataFrame[Any, Any, Any, Any, Any]"
NativePySpark: TypeAlias = _PySparkDataFrame
NativePySparkConnect: TypeAlias = _PySparkDataFrame
NativeSparkLike: TypeAlias = "NativeSQLFrame | NativePySpark | NativePySparkConnect"
NativeKnown: TypeAlias = "NativePolars | NativeArrow | NativePandasLike | NativeSparkLike | NativeDuckDB | NativeDask | NativeIbis"
NativeUnknown: TypeAlias = "NativeDataFrame | NativeSeries | NativeLazyFrame"
NativeAny: TypeAlias = "NativeKnown | NativeUnknown"
