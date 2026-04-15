from __future__ import annotations

import os
import uuid
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, cast

from narwhals._utils import generate_temporary_column_name
from narwhals.testing.constructors._name import ConstructorName

if TYPE_CHECKING:
    import ibis
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from ibis.backends.duckdb import Backend as IbisDuckDBBackend
    from pyspark.sql import SparkSession
    from sqlframe.duckdb import DuckDBSession

    from narwhals._native import NativeDask, NativeDuckDB, NativePySpark, NativeSQLFrame
    from narwhals.testing.typing import Data
    from narwhals.typing import IntoDataFrame, IntoFrame, IntoLazyFrame


def sqlframe_session() -> DuckDBSession:
    """Return a fresh in-memory `sqlframe` DuckDB session."""
    from sqlframe.duckdb import DuckDBSession

    # NOTE: `__new__` override inferred by `pyright` only
    # https://github.com/eakmanrq/sqlframe/blob/772b3a6bfe5a1ffd569b7749d84bea2f3a314510/sqlframe/base/session.py#L181-L184
    return cast("DuckDBSession", DuckDBSession())  # type: ignore[redundant-cast]


def pyspark_session() -> SparkSession:  # pragma: no cover
    """Return a singleton local `pyspark` (or pyspark[connect]) session."""
    if is_spark_connect := os.environ.get("SPARK_CONNECT", None):
        from pyspark.sql.connect.session import SparkSession
    else:
        from pyspark.sql import SparkSession
    builder = cast("SparkSession.Builder", SparkSession.builder).appName("unit-tests")
    builder = (
        builder.remote(f"sc://localhost:{os.environ.get('SPARK_PORT', '15002')}")
        if is_spark_connect
        else builder.master("local[1]").config("spark.ui.enabled", "false")
    )
    return (
        builder.config("spark.default.parallelism", "1")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


@lru_cache(maxsize=1)
def _ibis_backend() -> IbisDuckDBBackend:  # pragma: no cover
    """Cached singleton in-memory ibis backend, so all tables share one database."""
    import ibis

    return ibis.duckdb.connect()


@lru_cache(maxsize=1)
def _pyspark_session_lazy() -> SparkSession:  # pragma: no cover
    """Cached pyspark session; created on first use, stopped at interpreter exit."""
    from atexit import register

    with warnings.catch_warnings():
        # The spark session seems to trigger a polars warning.
        warnings.filterwarnings(
            "ignore", r"Using fork\(\) can cause Polars", category=RuntimeWarning
        )
        session = pyspark_session()
        register(session.stop)
        return session


class ConstructorBase(ABC):
    """Abstract base for any constructor exposed by `narwhals.testing`.

    A constructor is a callable that turns a column-oriented `dict` (typed as
    [`Data`][narwhals.testing.typing.Data]) into a native dataframe / lazy frame,
    plus a typed [`ConstructorName`][] that identifies the backend.
    """

    name: ClassVar[ConstructorName]

    @abstractmethod
    def __call__(self, obj: Data) -> IntoFrame:
        """Build a native frame from `obj`."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __hash__(self) -> int:
        return hash((type(self), self.name))

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is type(other) and self.name == cast("ConstructorBase", other).name
        )


class ConstructorEagerBase(ConstructorBase):
    """A constructor that returns an *eager* native dataframe."""

    @abstractmethod
    def __call__(self, obj: Data) -> IntoDataFrame: ...


class ConstructorLazyBase(ConstructorBase):
    """A constructor that returns a *lazy* native frame."""

    @abstractmethod
    def __call__(self, obj: Data) -> IntoLazyFrame: ...


# --- Eager constructors ------------------------------------------------------


class PandasConstructor(ConstructorEagerBase):
    name = ConstructorName.PANDAS

    def __call__(self, obj: Data) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(obj)


class PandasNullableConstructor(ConstructorEagerBase):
    name = ConstructorName.PANDAS_NULLABLE

    def __call__(self, obj: Data) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(obj).convert_dtypes(dtype_backend="numpy_nullable")


class PandasPyArrowConstructor(ConstructorEagerBase):
    name = ConstructorName.PANDAS_PYARROW

    def __call__(self, obj: Data) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(obj).convert_dtypes(dtype_backend="pyarrow")


class PyArrowConstructor(ConstructorEagerBase):
    name = ConstructorName.PYARROW

    def __call__(self, obj: Data) -> pa.Table:
        import pyarrow as pa

        return pa.table(obj)  # type:ignore[arg-type]


class ModinConstructor(ConstructorEagerBase):  # pragma: no cover
    name = ConstructorName.MODIN

    def __call__(self, obj: Data) -> IntoDataFrame:
        import modin.pandas as mpd
        import pandas as pd

        return cast("IntoDataFrame", mpd.DataFrame(pd.DataFrame(obj)))


class ModinPyArrowConstructor(ConstructorEagerBase):
    name = ConstructorName.MODIN_PYARROW

    def __call__(self, obj: Data) -> IntoDataFrame:
        import modin.pandas as mpd
        import pandas as pd

        df = mpd.DataFrame(pd.DataFrame(obj)).convert_dtypes(dtype_backend="pyarrow")
        return cast("IntoDataFrame", df)


class CudfConstructor(ConstructorEagerBase):  # pragma: no cover
    name = ConstructorName.CUDF

    def __call__(self, obj: Data) -> IntoDataFrame:
        import cudf

        return cast("IntoDataFrame", cudf.DataFrame(obj))


class PolarsEagerConstructor(ConstructorEagerBase):
    name = ConstructorName.POLARS_EAGER

    def __call__(self, obj: Data) -> pl.DataFrame:
        import polars as pl

        return pl.DataFrame(obj)


# --- Lazy constructors -------------------------------------------------------


class PolarsLazyConstructor(ConstructorLazyBase):
    name = ConstructorName.POLARS_LAZY

    def __call__(self, obj: Data) -> pl.LazyFrame:
        import polars as pl

        return pl.LazyFrame(obj)


class DaskConstructor(ConstructorLazyBase):  # pragma: no cover
    name = ConstructorName.DASK

    def __init__(self, npartitions: int = 2) -> None:
        self.npartitions = npartitions

    def __call__(self, obj: Data) -> NativeDask:
        import dask.dataframe as dd

        return cast("NativeDask", dd.from_dict(obj, npartitions=self.npartitions))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(npartitions={self.npartitions})"

    def __hash__(self) -> int:
        return hash((type(self), self.name, self.npartitions))

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is type(other)
            and self.npartitions == cast("DaskConstructor", other).npartitions
        )


class DuckDBConstructor(ConstructorLazyBase):
    name = ConstructorName.DUCKDB

    def __call__(self, obj: Data) -> NativeDuckDB:
        import duckdb
        import pyarrow as pa

        duckdb.sql("""set timezone = 'UTC'""")
        _df = pa.table(obj)  # type:ignore[arg-type]
        return duckdb.sql("select * from _df")


class PySparkConstructor(ConstructorLazyBase):  # pragma: no cover
    name = ConstructorName.PYSPARK

    def __call__(self, obj: Data) -> NativePySpark:
        session = _pyspark_session_lazy()
        _obj = deepcopy(obj)
        index_col_name = generate_temporary_column_name(n_bytes=8, columns=list(_obj))
        _obj[index_col_name] = list(range(len(_obj[next(iter(_obj))])))
        result = (
            session.createDataFrame([*zip(*_obj.values())], schema=[*_obj.keys()])
            .repartition(2)
            .orderBy(index_col_name)
            .drop(index_col_name)
        )
        return cast("NativePySpark", result)


class PySparkConnectConstructor(PySparkConstructor):  # pragma: no cover
    name = ConstructorName.PYSPARK_CONNECT


class SQLFrameConstructor(ConstructorLazyBase):
    name = ConstructorName.SQLFRAME

    def __call__(self, obj: Data) -> NativeSQLFrame:
        session = sqlframe_session()
        return session.createDataFrame([*zip(*obj.values())], schema=[*obj.keys()])


class IbisConstructor(ConstructorLazyBase):
    name = ConstructorName.IBIS

    def __call__(self, obj: Data) -> ibis.Table:
        import pyarrow as pa

        table = pa.table(obj)  # type:ignore[arg-type]
        table_name = str(uuid.uuid4())
        return _ibis_backend().create_table(table_name, table)


_ALL_CONSTRUCTORS: dict[ConstructorName, ConstructorBase] = {
    ConstructorName.PANDAS: PandasConstructor(),
    ConstructorName.PANDAS_NULLABLE: PandasNullableConstructor(),
    ConstructorName.PANDAS_PYARROW: PandasPyArrowConstructor(),
    ConstructorName.PYARROW: PyArrowConstructor(),
    ConstructorName.MODIN: ModinConstructor(),
    ConstructorName.MODIN_PYARROW: ModinPyArrowConstructor(),
    ConstructorName.CUDF: CudfConstructor(),
    ConstructorName.POLARS_EAGER: PolarsEagerConstructor(),
    ConstructorName.POLARS_LAZY: PolarsLazyConstructor(),
    ConstructorName.DASK: DaskConstructor(),
    ConstructorName.DUCKDB: DuckDBConstructor(),
    ConstructorName.PYSPARK: PySparkConstructor(),
    ConstructorName.PYSPARK_CONNECT: PySparkConnectConstructor(),
    ConstructorName.SQLFRAME: SQLFrameConstructor(),
    ConstructorName.IBIS: IbisConstructor(),
}

_BACKEND_REQUIREMENTS: dict[ConstructorName, tuple[str, ...]] = {
    ConstructorName.PANDAS: ("pandas",),
    ConstructorName.PANDAS_NULLABLE: ("pandas",),
    ConstructorName.PANDAS_PYARROW: ("pandas", "pyarrow"),
    ConstructorName.PYARROW: ("pyarrow",),
    ConstructorName.MODIN: ("modin",),
    ConstructorName.MODIN_PYARROW: ("modin", "pyarrow"),
    ConstructorName.CUDF: ("cudf",),
    ConstructorName.POLARS_EAGER: ("polars",),
    ConstructorName.POLARS_LAZY: ("polars",),
    ConstructorName.DASK: ("dask",),
    ConstructorName.DUCKDB: ("duckdb", "pyarrow"),
    ConstructorName.PYSPARK: ("pyspark",),
    ConstructorName.PYSPARK_CONNECT: ("pyspark",),
    ConstructorName.SQLFRAME: ("sqlframe", "duckdb"),
    ConstructorName.IBIS: ("ibis", "duckdb", "pyarrow"),
}


__all__ = [
    "ConstructorBase",
    "ConstructorEagerBase",
    "ConstructorLazyBase",
    "CudfConstructor",
    "DaskConstructor",
    "DuckDBConstructor",
    "IbisConstructor",
    "ModinConstructor",
    "ModinPyArrowConstructor",
    "PandasConstructor",
    "PandasNullableConstructor",
    "PandasPyArrowConstructor",
    "PolarsEagerConstructor",
    "PolarsLazyConstructor",
    "PyArrowConstructor",
    "PySparkConnectConstructor",
    "PySparkConstructor",
    "SQLFrameConstructor",
    "pyspark_session",
    "sqlframe_session",
]
