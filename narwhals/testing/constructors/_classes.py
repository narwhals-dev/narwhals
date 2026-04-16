"""Concrete constructor classes and auto-registration machinery.

Each constructor wraps one backend library (pandas, Polars, DuckDB, ...)
and knows how to turn a column-oriented `dict` into a native frame.

## Adding a new constructor

1. Choose the right base class:

    * `ConstructorEagerBase`: if the backend returns an eager dataframe.
    * `ConstructorLazyBase`: if the backend returns a lazy frame.

2. Add a member to `ConstructorName` in `_name.py` and register the corresponding
    `Implementation` mapping in `_NAME_TO_IMPL`.

3. Define the class in this module. Specify:

    * `requirements`: the packages that `importlib.util.find_spec` should check
    * `legacy_name` (if relevant): the old `str(constructor)` value used in existing
        tests as keyword arguments in the class header:

    ```py
    class MyBackendConstructor(
        ConstructorLazyBase,
        requirements=("my_backend",),
        legacy_name="my_backend_lazy_constructor",
    ):
        name = ConstructorName.MY_BACKEND

        def __call__(self, obj: Data, /, **kwds: Any) -> ...:
            import my_backend

            return my_backend.from_dict(obj)
    ```

That is all. `__init_subclass__` on `ConstructorBase` will automatically register
a default singleton into `_registry`, record the *requirements*, and store the *legacy_name*.
"""

from __future__ import annotations

import os
import uuid
import warnings
from copy import deepcopy
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

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


class ConstructorBase(Protocol):
    """Abstract base for any constructor exposed by `narwhals.testing`.

    A constructor is a callable that turns a column-oriented `dict` (typed as
    [`Data`][narwhals.testing.typing.Data]) into a native dataframe / lazy frame,
    plus a typed [`ConstructorName`][] that identifies the backend.

    Subclasses are automatically registered when they set ``name`` as a
    class variable and pass ``requirements`` / ``legacy_name`` as
    keyword arguments in the class definition.
    """

    _registry: ClassVar[dict[ConstructorName, ConstructorBase]] = {}
    _requirements: ClassVar[dict[ConstructorName, tuple[str, ...]]] = {}
    _legacy_names: ClassVar[dict[ConstructorName, str]] = {}

    name: ClassVar[ConstructorName]

    def __init_subclass__(
        cls, *, requirements: tuple[str, ...] = (), legacy_name: str = "", **kwargs: Any
    ) -> None:
        """Register concrete subclasses automatically.

        Arguments:
            requirements: Package names that must be importable for
                this constructor to be available (checked via
                ``importlib.util.find_spec``).
            legacy_name: Value returned by ``str(constructor)`` for
                backward compatibility with test assertions that
                match on the old naming scheme.
            **kwargs: Forwarded to ``super().__init_subclass__``.
        """
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            return
        instance = cls()
        ConstructorBase._registry[cls.name] = instance
        ConstructorBase._requirements[cls.name] = requirements
        ConstructorBase._legacy_names[cls.name] = legacy_name

    def __call__(self, obj: Data, /, **kwds: Any) -> IntoFrame:
        """Build a native frame from `obj`."""
        ...

    @property
    def identifier(self) -> str:
        """Instance-level string identifier for test IDs."""
        return str(self.name)

    def __str__(self) -> str:
        # NOTE: This is a temporary hack
        # TODO(Unassigned): Remove once all the
        # `"backend" in str(constructor)` statements in the
        # test suite are properly replaced
        return _LEGACY_NAME[self.name]

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

    def __call__(self, obj: Data, /, **kwds: Any) -> IntoDataFrame: ...


class ConstructorLazyBase(ConstructorBase):
    """A constructor that returns a *lazy* native frame."""

    def __call__(self, obj: Data, /, **kwds: Any) -> IntoLazyFrame: ...


# Eager constructors


class PandasConstructor(
    ConstructorEagerBase, requirements=("pandas",), legacy_name="pandas_constructor"
):
    """Constructor backed by ``pandas.DataFrame`` with default NumPy dtypes."""

    name = ConstructorName.PANDAS

    def __call__(self, obj: Data, /, **kwds: Any) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(obj, **kwds)


class PandasNullableConstructor(
    ConstructorEagerBase,
    requirements=("pandas",),
    legacy_name="pandas_nullable_constructor",
):
    """Constructor backed by ``pandas.DataFrame`` with ``numpy_nullable`` dtypes."""

    name = ConstructorName.PANDAS_NULLABLE

    def __call__(self, obj: Data, /, **kwds: Any) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(obj, **kwds).convert_dtypes(dtype_backend="numpy_nullable")


class PandasPyArrowConstructor(
    ConstructorEagerBase,
    requirements=("pandas", "pyarrow"),
    legacy_name="pandas_pyarrow_constructor",
):
    """Constructor backed by ``pandas.DataFrame`` with ``pyarrow`` dtypes."""

    name = ConstructorName.PANDAS_PYARROW

    def __call__(self, obj: Data, /, **kwds: Any) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(obj, **kwds).convert_dtypes(dtype_backend="pyarrow")


class PyArrowConstructor(
    ConstructorEagerBase,
    requirements=("pyarrow",),
    legacy_name="pyarrow_table_constructor",
):
    """Constructor backed by ``pyarrow.Table``."""

    name = ConstructorName.PYARROW

    def __call__(self, obj: Data, /, **kwds: Any) -> pa.Table:
        import pyarrow as pa

        return pa.table(obj, **kwds)  # type:ignore[arg-type]


class ModinConstructor(
    ConstructorEagerBase, requirements=("modin",), legacy_name="modin_constructor"
):  # pragma: no cover
    """Constructor backed by ``modin.pandas.DataFrame``."""

    name = ConstructorName.MODIN

    def __call__(self, obj: Data, /, **kwds: Any) -> IntoDataFrame:
        import modin.pandas as mpd
        import pandas as pd

        return cast("IntoDataFrame", mpd.DataFrame(pd.DataFrame(obj, **kwds)))


class ModinPyArrowConstructor(
    ConstructorEagerBase,
    requirements=("modin", "pyarrow"),
    legacy_name="modin_pyarrow_constructor",
):
    """Constructor backed by ``modin.pandas.DataFrame`` with ``pyarrow`` dtypes."""

    name = ConstructorName.MODIN_PYARROW

    def __call__(self, obj: Data, /, **kwds: Any) -> IntoDataFrame:
        import modin.pandas as mpd
        import pandas as pd

        df = mpd.DataFrame(pd.DataFrame(obj, **kwds)).convert_dtypes(
            dtype_backend="pyarrow"
        )
        return cast("IntoDataFrame", df)


class CudfConstructor(
    ConstructorEagerBase, requirements=("cudf",), legacy_name="cudf_constructor"
):  # pragma: no cover
    """Constructor backed by ``cudf.DataFrame`` (requires GPU)."""

    name = ConstructorName.CUDF

    def __call__(self, obj: Data, /, **kwds: Any) -> IntoDataFrame:
        import cudf

        return cast("IntoDataFrame", cudf.DataFrame(obj, **kwds))


class PolarsEagerConstructor(
    ConstructorEagerBase, requirements=("polars",), legacy_name="polars_eager_constructor"
):
    """Constructor backed by ``polars.DataFrame``."""

    name = ConstructorName.POLARS_EAGER

    def __call__(self, obj: Data, /, **kwds: Any) -> pl.DataFrame:
        import polars as pl

        return pl.DataFrame(obj, **kwds)


# Lazy constructors


class PolarsLazyConstructor(
    ConstructorLazyBase, requirements=("polars",), legacy_name="polars_lazy_constructor"
):
    """Constructor backed by ``polars.LazyFrame``."""

    name = ConstructorName.POLARS_LAZY

    def __call__(self, obj: Data, /, **kwds: Any) -> pl.LazyFrame:
        import polars as pl

        return pl.LazyFrame(obj, **kwds)


class DaskConstructor(
    ConstructorLazyBase, requirements=("dask",), legacy_name="dask_lazy_p2_constructor"
):  # pragma: no cover
    """Constructor backed by ``dask.dataframe``.

    Arguments:
        npartitions: Number of Dask partitions (default ``1``).
    """

    name = ConstructorName.DASK

    def __init__(self, npartitions: int = 2) -> None:
        self.npartitions = npartitions

    def __call__(self, obj: Data, /, **kwds: Any) -> NativeDask:
        import dask.dataframe as dd

        return cast("NativeDask", dd.from_dict(obj, npartitions=self.npartitions, **kwds))

    @property
    def identifier(self) -> str:
        """Identifier that encodes the number of partitions."""
        return f"dask[p{self.npartitions}]"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(npartitions={self.npartitions})"

    def __hash__(self) -> int:
        return hash((type(self), self.name, self.npartitions))

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is type(other)
            and self.npartitions == cast("DaskConstructor", other).npartitions
        )


class DuckDBConstructor(
    ConstructorLazyBase,
    requirements=("duckdb", "pyarrow"),
    legacy_name="duckdb_lazy_constructor",
):
    """Constructor backed by DuckDB (via ``duckdb.sql``)."""

    name = ConstructorName.DUCKDB

    def __call__(self, obj: Data, /, **kwds: Any) -> NativeDuckDB:
        import duckdb
        import pyarrow as pa

        duckdb.sql("""set timezone = 'UTC'""")
        _df = pa.table(obj, **kwds)  # type:ignore[arg-type]
        return duckdb.sql("select * from _df")


class PySparkConstructor(
    ConstructorLazyBase, requirements=("pyspark",), legacy_name="pyspark_lazy_constructor"
):  # pragma: no cover
    """Constructor backed by ``pyspark.sql.DataFrame``."""

    name = ConstructorName.PYSPARK

    def __call__(self, obj: Data, /, **kwds: Any) -> NativePySpark:
        session = _pyspark_session_lazy()
        _obj = deepcopy(obj)
        index_col_name = generate_temporary_column_name(n_bytes=8, columns=list(_obj))
        _obj[index_col_name] = list(range(len(_obj[next(iter(_obj))])))
        result = (
            session.createDataFrame([*zip(*_obj.values())], schema=[*_obj.keys()], **kwds)
            .repartition(2)
            .orderBy(index_col_name)
            .drop(index_col_name)
        )
        return cast("NativePySpark", result)


class PySparkConnectConstructor(
    PySparkConstructor, requirements=("pyspark",), legacy_name="pyspark_lazy_constructor"
):  # pragma: no cover
    """Constructor backed by PySpark Connect (Spark Connect protocol)."""

    name = ConstructorName.PYSPARK_CONNECT


class SQLFrameConstructor(
    ConstructorLazyBase,
    requirements=("sqlframe", "duckdb"),
    legacy_name="sqlframe_pyspark_lazy_constructor",
):
    """Constructor backed by ``sqlframe`` (DuckDB session)."""

    name = ConstructorName.SQLFRAME

    def __call__(self, obj: Data, /, **kwds: Any) -> NativeSQLFrame:
        session = sqlframe_session()
        return session.createDataFrame(
            [*zip(*obj.values())], schema=[*obj.keys()], **kwds
        )


class IbisConstructor(
    ConstructorLazyBase,
    requirements=("ibis", "duckdb", "pyarrow"),
    legacy_name="ibis_lazy_constructor",
):
    """Constructor backed by ``ibis`` (DuckDB backend)."""

    name = ConstructorName.IBIS

    def __call__(self, obj: Data, /, **kwds: Any) -> ibis.Table:
        import pyarrow as pa

        table = pa.table(obj)  # type:ignore[arg-type]
        table_name = str(uuid.uuid4())
        return _ibis_backend().create_table(table_name, table, **kwds)


# TODO(Unassigned): Remove once all the `"backend" in str(constructor)`
# statements in the test suite are properly replaced.
_LEGACY_NAME = ConstructorBase._legacy_names
