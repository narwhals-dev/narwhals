"""Concrete constructor classes and auto-registration machinery.

Each constructor wraps one backend library (pandas, Polars, DuckDB, ...)
and knows how to turn a column-oriented `dict` into a native frame.

All static metadata for a backend lives on its constructor class, colocated
with the `__call__` implementation. Adding a constructor is a one-step
declaration — the `__init_subclass__` hook then auto-registers a singleton.

## Adding a new constructor

1. Choose the right base class:

    * `ConstructorEagerBase`: if the backend returns an eager dataframe.
    * `ConstructorLazyBase`: if the backend returns a lazy frame.

2. Define the class in this module and declare its metadata in the class
    header as keyword arguments:

    ```py
    class MyBackendConstructor(
        ConstructorLazyBase,
        implementation=Implementation.MY_BACKEND,
        requirements=("my_backend",),
        legacy_name="my_backend_lazy_constructor",
    ):
        name = "my_backend"

        def __call__(self, obj: Data, /, **kwds: Any) -> ...:
            import my_backend

            return my_backend.from_dict(obj)
    ```

That is all. `__init_subclass__` on `ConstructorBase` automatically registers
a default singleton into `_registry`, keyed by the string `name`.
"""

from __future__ import annotations

import os
import uuid
import warnings
from copy import deepcopy
from functools import lru_cache
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from narwhals._utils import Implementation, generate_temporary_column_name

if TYPE_CHECKING:
    from collections.abc import Iterable

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


__all__ = (
    "ALL_CONSTRUCTORS",
    "ALL_CPU_CONSTRUCTORS",
    "DEFAULT_CONSTRUCTORS",
    "ConstructorBase",
    "ConstructorEagerBase",
    "available_constructors",
    "get_constructor",
    "is_backend_available",
    "prepare_constructors",
    "pyspark_session",
    "sqlframe_session",
)


class ConstructorBase(Protocol):
    """Abstract base for any constructor exposed by `narwhals.testing`.

    A constructor is a callable that turns a column-oriented `dict` (typed as
    [`Data`][narwhals.testing.typing.Data]) into a native dataframe / lazy frame,
    plus a string `name` that identifies the backend (e.g. `"pandas[pyarrow]"`).

    Subclasses declare their backend metadata (implementation, requirements,
    legacy name, nullability, GPU need) as keyword arguments in the class
    header. `__init_subclass__` stores those on the class and registers a
    default singleton into `_registry`.
    """

    _registry: ClassVar[dict[str, ConstructorBase]] = {}

    name: ClassVar[str]
    implementation: ClassVar[Implementation]
    requirements: ClassVar[tuple[str, ...]] = ()
    legacy_name: ClassVar[str] = ""
    is_eager: ClassVar[bool] = False
    is_non_nullable: ClassVar[bool] = False
    needs_gpu: ClassVar[bool] = False

    def __init_subclass__(
        cls,
        *,
        implementation: Implementation | None = None,
        requirements: tuple[str, ...] = (),
        legacy_name: str = "",
        is_non_nullable: bool = False,
        needs_gpu: bool = False,
        **kwargs: Any,
    ) -> None:
        """Register concrete subclasses automatically.

        Arguments:
            implementation: The [`Implementation`][] this constructor belongs to.
            requirements: Package names that must be importable for this constructor
                to be available (checked via `importlib.util.find_spec`).
            legacy_name: Value returned by `str(constructor)` for backward compatibility
                with test assertions that match on the old naming scheme.
            is_non_nullable: Whether the backend lacks native null support.
            needs_gpu: Whether the backend requires GPU hardware.
            **kwargs: Forwarded to `super().__init_subclass__`.
        """
        super().__init_subclass__(**kwargs)
        if implementation is not None:
            cls.implementation = implementation

        cls.requirements = requirements
        cls.legacy_name = legacy_name
        cls.is_non_nullable = is_non_nullable
        cls.needs_gpu = needs_gpu

        if "name" not in cls.__dict__:
            return
        if not hasattr(cls, "implementation"):
            msg = (
                f"Constructor {cls.__name__} is missing `implementation` "
                "kwarg in its class header."
            )
            raise TypeError(msg)
        ConstructorBase._registry[cls.name] = cls()

    def __call__(self, obj: Data, /, **kwds: Any) -> IntoFrame:
        """Build a native frame from `obj`."""
        ...

    @property
    def identifier(self) -> str:
        """Instance-level string identifier for test IDs."""
        return self.name

    @property
    def is_lazy(self) -> bool:
        """Whether this constructor produces a lazy native frame."""
        return not self.is_eager

    @property
    def is_pandas(self) -> bool:
        """Whether this is one of the pandas constructors."""
        return self.implementation.is_pandas()

    @property
    def is_modin(self) -> bool:
        """Whether this is one of the modin constructors."""
        return self.implementation.is_modin()

    @property
    def is_cudf(self) -> bool:
        """Whether this is the cudf constructor."""
        return self.implementation.is_cudf()

    @property
    def is_pandas_like(self) -> bool:
        """Whether this constructor produces a pandas-like dataframe (pandas, modin, cudf)."""
        return self.implementation.is_pandas_like()

    @property
    def is_polars(self) -> bool:
        """Whether this is one of the polars constructors."""
        return self.implementation.is_polars()

    @property
    def is_pyarrow(self) -> bool:
        """Whether this is the pyarrow table constructor."""
        return self.implementation.is_pyarrow()

    @property
    def is_dask(self) -> bool:
        """Whether this is the dask constructor."""
        return self.implementation.is_dask()

    @property
    def is_duckdb(self) -> bool:
        """Whether this is the duckdb constructor."""
        return self.implementation.is_duckdb()

    @property
    def is_pyspark(self) -> bool:
        """Whether this is one of the pyspark constructors."""
        impl = self.implementation
        return impl.is_pyspark() or impl.is_pyspark_connect()

    @property
    def is_sqlframe(self) -> bool:
        """Whether this is the sqlframe constructor."""
        return self.implementation.is_sqlframe()

    @property
    def is_ibis(self) -> bool:
        """Whether this is the ibis constructor."""
        return self.implementation.is_ibis()

    @property
    def is_spark_like(self) -> bool:
        """Whether this constructor uses a spark-like backend (pyspark, sqlframe)."""
        return self.implementation.is_spark_like()

    @property
    def needs_pyarrow(self) -> bool:
        """Whether this constructor requires `pyarrow` to be installed."""
        return "pyarrow" in self.requirements

    @property
    def is_available(self) -> bool:
        """Whether every package this constructor needs is importable."""
        return is_backend_available(*self.requirements)

    def __str__(self) -> str:
        # NOTE: This is a temporary hack
        # TODO(Unassigned): Remove once all the `"backend" in str(constructor)`
        # statements in the test suite are properly replaced
        return self.legacy_name

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

    is_eager: ClassVar[bool] = True

    def __call__(self, obj: Data, /, **kwds: Any) -> IntoDataFrame: ...


class ConstructorLazyBase(ConstructorBase):
    """A constructor that returns a *lazy* native frame."""

    is_eager: ClassVar[bool] = False

    def __call__(self, obj: Data, /, **kwds: Any) -> IntoLazyFrame: ...


# Eager constructors


class PandasConstructor(
    ConstructorEagerBase,
    implementation=Implementation.PANDAS,
    requirements=("pandas",),
    legacy_name="pandas_constructor",
    is_non_nullable=True,
):
    """Constructor backed by `pandas.DataFrame` with default NumPy dtypes."""

    name = "pandas"

    def __call__(self, obj: Data, /, **kwds: Any) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(obj, **kwds)


class PandasNullableConstructor(
    ConstructorEagerBase,
    implementation=Implementation.PANDAS,
    requirements=("pandas",),
    legacy_name="pandas_nullable_constructor",
):
    """Constructor backed by `pandas.DataFrame` with `numpy_nullable` dtypes."""

    name = "pandas[nullable]"

    def __call__(self, obj: Data, /, **kwds: Any) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(obj, **kwds).convert_dtypes(dtype_backend="numpy_nullable")


class PandasPyArrowConstructor(
    ConstructorEagerBase,
    implementation=Implementation.PANDAS,
    requirements=("pandas", "pyarrow"),
    legacy_name="pandas_pyarrow_constructor",
):
    """Constructor backed by `pandas.DataFrame` with `pyarrow` dtypes."""

    name = "pandas[pyarrow]"

    def __call__(self, obj: Data, /, **kwds: Any) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(obj, **kwds).convert_dtypes(dtype_backend="pyarrow")


class PyArrowConstructor(
    ConstructorEagerBase,
    implementation=Implementation.PYARROW,
    requirements=("pyarrow",),
    legacy_name="pyarrow_table_constructor",
):
    """Constructor backed by `pyarrow.Table`."""

    name = "pyarrow"

    def __call__(self, obj: Data, /, **kwds: Any) -> pa.Table:
        import pyarrow as pa

        return pa.table(obj, **kwds)  # type:ignore[arg-type]


class ModinConstructor(
    ConstructorEagerBase,
    implementation=Implementation.MODIN,
    requirements=("modin",),
    legacy_name="modin_constructor",
    is_non_nullable=True,
):  # pragma: no cover
    """Constructor backed by `modin.pandas.DataFrame`."""

    name = "modin"

    def __call__(self, obj: Data, /, **kwds: Any) -> IntoDataFrame:
        import modin.pandas as mpd
        import pandas as pd

        return cast("IntoDataFrame", mpd.DataFrame(pd.DataFrame(obj, **kwds)))


class ModinPyArrowConstructor(
    ConstructorEagerBase,
    implementation=Implementation.MODIN,
    requirements=("modin", "pyarrow"),
    legacy_name="modin_pyarrow_constructor",
):
    """Constructor backed by `modin.pandas.DataFrame` with `pyarrow` dtypes."""

    name = "modin[pyarrow]"

    def __call__(self, obj: Data, /, **kwds: Any) -> IntoDataFrame:
        import modin.pandas as mpd
        import pandas as pd

        df = mpd.DataFrame(pd.DataFrame(obj, **kwds)).convert_dtypes(
            dtype_backend="pyarrow"
        )
        return cast("IntoDataFrame", df)


class CudfConstructor(
    ConstructorEagerBase,
    implementation=Implementation.CUDF,
    requirements=("cudf",),
    legacy_name="cudf_constructor",
    needs_gpu=True,
):  # pragma: no cover
    """Constructor backed by `cudf.DataFrame` (requires GPU)."""

    name = "cudf"

    def __call__(self, obj: Data, /, **kwds: Any) -> IntoDataFrame:
        import cudf

        return cast("IntoDataFrame", cudf.DataFrame(obj, **kwds))


class PolarsEagerConstructor(
    ConstructorEagerBase,
    implementation=Implementation.POLARS,
    requirements=("polars",),
    legacy_name="polars_eager_constructor",
):
    """Constructor backed by `polars.DataFrame`."""

    name = "polars[eager]"

    def __call__(self, obj: Data, /, **kwds: Any) -> pl.DataFrame:
        import polars as pl

        return pl.DataFrame(obj, **kwds)


# Lazy constructors


class PolarsLazyConstructor(
    ConstructorLazyBase,
    implementation=Implementation.POLARS,
    requirements=("polars",),
    legacy_name="polars_lazy_constructor",
):
    """Constructor backed by `polars.LazyFrame`."""

    name = "polars[lazy]"

    def __call__(self, obj: Data, /, **kwds: Any) -> pl.LazyFrame:
        import polars as pl

        return pl.LazyFrame(obj, **kwds)


class DaskConstructor(
    ConstructorLazyBase,
    implementation=Implementation.DASK,
    requirements=("dask",),
    legacy_name="dask_lazy_p2_constructor",
    is_non_nullable=True,
):  # pragma: no cover
    """Constructor backed by `dask.dataframe`.

    Arguments:
        npartitions: Number of Dask partitions (default `1`).
    """

    name = "dask"

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
    implementation=Implementation.DUCKDB,
    requirements=("duckdb", "pyarrow"),
    legacy_name="duckdb_lazy_constructor",
):
    """Constructor backed by DuckDB (via `duckdb.sql`)."""

    name = "duckdb"

    def __call__(self, obj: Data, /, **kwds: Any) -> NativeDuckDB:
        import duckdb
        import pyarrow as pa

        duckdb.sql("""set timezone = 'UTC'""")
        _df = pa.table(obj, **kwds)  # type:ignore[arg-type]
        return duckdb.sql("select * from _df")


class PySparkConstructor(
    ConstructorLazyBase,
    implementation=Implementation.PYSPARK,
    requirements=("pyspark",),
    legacy_name="pyspark_lazy_constructor",
):  # pragma: no cover
    """Constructor backed by `pyspark.sql.DataFrame`."""

    name = "pyspark"

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
    PySparkConstructor,
    implementation=Implementation.PYSPARK_CONNECT,
    requirements=("pyspark",),
    legacy_name="pyspark_lazy_constructor",
):  # pragma: no cover
    """Constructor backed by PySpark Connect (Spark Connect protocol)."""

    name = "pyspark[connect]"


class SQLFrameConstructor(
    ConstructorLazyBase,
    implementation=Implementation.SQLFRAME,
    requirements=("sqlframe", "duckdb"),
    legacy_name="sqlframe_pyspark_lazy_constructor",
):
    """Constructor backed by `sqlframe` (DuckDB session)."""

    name = "sqlframe"

    def __call__(self, obj: Data, /, **kwds: Any) -> NativeSQLFrame:
        session = sqlframe_session()
        return session.createDataFrame(
            [*zip(*obj.values())], schema=[*obj.keys()], **kwds
        )


class IbisConstructor(
    ConstructorLazyBase,
    implementation=Implementation.IBIS,
    requirements=("ibis", "duckdb", "pyarrow"),
    legacy_name="ibis_lazy_constructor",
):
    """Constructor backed by `ibis` (DuckDB backend)."""

    name = "ibis"

    def __call__(self, obj: Data, /, **kwds: Any) -> ibis.Table:
        import pyarrow as pa

        table = pa.table(obj)  # type:ignore[arg-type]
        table_name = str(uuid.uuid4())
        return _ibis_backend().create_table(table_name, table, **kwds)


ALL_CONSTRUCTORS: dict[str, ConstructorBase] = ConstructorBase._registry
"""All registered constructors keyed by their string identifier."""

DEFAULT_CONSTRUCTORS: frozenset[str] = frozenset(
    {
        "pandas",
        "pandas[pyarrow]",
        "polars[eager]",
        "pyarrow",
        "duckdb",
        "sqlframe",
        "ibis",
    }
)
"""Subset of constructors enabled by default for parametrised tests when the
user does not pass `--constructors` (mirrors the historical Narwhals defaults).
"""

ALL_CPU_CONSTRUCTORS: frozenset[str] = frozenset(
    name for name, c in ConstructorBase._registry.items() if not c.needs_gpu
)
"""All constructors that do not require GPU hardware."""


def available_constructors() -> frozenset[str]:
    """Return the names of every constructor whose backend is importable.

    Examples:
        >>> from narwhals.testing.constructors import available_constructors
        >>> "pandas" in available_constructors()
        True
    """
    return frozenset(name for name, c in ALL_CONSTRUCTORS.items() if c.is_available)


def get_constructor(name: str) -> ConstructorBase:
    """Return the registered singleton constructor for `name`.

    Arguments:
        name: The string identifier of a registered constructor
            (e.g. `"pandas[pyarrow]"`).

    Raises:
        ValueError: If `name` is not a registered constructor identifier.

    Examples:
        >>> from narwhals.testing.constructors import get_constructor
        >>> get_constructor("pandas")
        PandasConstructor()
    """
    try:
        return ALL_CONSTRUCTORS[name]
    except KeyError as exc:
        valid = sorted(ALL_CONSTRUCTORS)
        msg = f"Unknown constructor {name!r}. Expected one of: {valid}."
        raise ValueError(msg) from exc


def prepare_constructors(
    *, include: Iterable[str] | None = None, exclude: Iterable[str] | None = None
) -> list[ConstructorBase]:
    """Return available constructors, optionally filtered.

    Arguments:
        include: If given, only return constructors whose name is in this set.
        exclude: If given, remove constructors whose name is in this set.

    Examples:
        >>> from narwhals.testing.constructors import prepare_constructors
        >>> constructors = prepare_constructors(include=["pandas", "polars[eager]"])
    """
    available = available_constructors()
    candidates: list[ConstructorBase] = [
        c for name, c in ALL_CONSTRUCTORS.items() if name in available
    ]
    if include is not None:
        inc = frozenset(include)
        candidates = [c for c in candidates if c.name in inc]
    if exclude is not None:
        exc = frozenset(exclude)
        candidates = [c for c in candidates if c.name not in exc]
    return sorted(candidates, key=lambda c: c.name)


def is_backend_available(*packages: str) -> bool:
    """Whether every package in `packages` can be imported in this environment.

    Examples:
        >>> from narwhals.testing.constructors import is_backend_available
        >>> is_backend_available("pandas")
        True
    """
    return all(find_spec(pkg) is not None for pkg in packages)


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
