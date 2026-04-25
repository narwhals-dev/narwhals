"""Constructor registry for `narwhals.testing`.

Each constructor wraps one backend library (pandas, Polars, DuckDB, ...) and
knows how to turn a column-oriented `dict` into a native frame.

Registration is explicit: wrap a plain builder function with `@frame_constructor.register(...)`.
The decorator instantiates a [`narwhals.testing.frame_constructor`][] with the
declared metadata and stores it in the shared `_registry`.

## Adding a new constructor

```py
from narwhals.testing import frame_constructor


@frame_constructor.register(
    name="my_backend",
    implementation=Implementation.MY_BACKEND,
    requirements=("my_backend",),
)
def my_backend_lazy_constructor(obj: Data, /, **kwds: Any) -> IntoLazyFrame:
    import my_backend

    return my_backend.from_dict(obj)
```
"""

from __future__ import annotations

import os
import uuid
import warnings
from copy import deepcopy
from functools import lru_cache
from importlib.util import find_spec
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)

from narwhals._utils import Implementation, generate_temporary_column_name

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

    import ibis
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from ibis.backends.duckdb import Backend as IbisDuckDBBackend
    from pyspark.sql import SparkSession
    from sqlframe.duckdb import DuckDBSession
    from typing_extensions import Concatenate, TypeAlias

    from narwhals import DataFrame, LazyFrame
    from narwhals._native import NativeDask, NativeDuckDB, NativePySpark, NativeSQLFrame
    from narwhals.testing.typing import Data
    from narwhals.typing import (
        IntoDataFrame,
        IntoDataFrameT,
        IntoFrame,
        IntoLazyFrame,
        IntoLazyFrameT,
    )


__all__ = (
    "available_backends",
    "available_cpu_backends",
    "frame_constructor",
    "get_backend_constructor",
    "is_backend_available",
    "prepare_backends",
    "pyspark_session",
    "sqlframe_session",
)

T_co = TypeVar("T_co", covariant=True, bound="IntoFrame")
R = TypeVar("R", bound="IntoFrame")


class frame_constructor(Generic[T_co]):  # noqa: N801
    """Callable wrapper around a backend frame builder.

    Turns a column-oriented `dict` (typed as [`Data`][narwhals.testing.typing.Data])
    into a native frame. Metadata (implementation, requirements, eager/lazy,
    nullability, GPU need) lives on the instance, alongside the wrapped
    `func`. Equality and hashing are keyed on `(type, name)`, so two lookups
    of the same registered constructor compare equal.

    Warning:
        Instances should be created via [`narwhals.testing.constructors.frame_constructor.register`][],
        which is the only supported entry point.

        Direct instantiation is allowed but **does not** register the instance.
    """

    _registry: ClassVar[dict[str, frame_constructor[IntoFrame]]] = {}

    func: Callable[Concatenate[Data, ...], T_co]

    def __init__(
        self,
        func: Callable[Concatenate[Data, ...], T_co],
        /,
        *,
        name: str,
        implementation: Implementation,
        requirements: tuple[str, ...] = (),
        is_eager: bool = False,
        is_nullable: bool = True,
        needs_gpu: bool = False,
    ) -> None:
        self.func = func
        self.name = name
        self.implementation = implementation
        self.requirements = requirements
        self.is_eager = is_eager
        self.is_nullable = is_nullable
        self.needs_gpu = needs_gpu

    @classmethod
    def register(
        cls,
        *,
        name: str,
        implementation: Implementation,
        requirements: tuple[str, ...] = (),
        is_eager: bool = False,
        is_nullable: bool = True,
        needs_gpu: bool = False,
    ) -> Callable[[Callable[Concatenate[Data, ...], R]], frame_constructor[R]]:
        """Decorator: register `func` as the constructor named `name`.

        Arguments:
            name: The string identifier of the constructor (e.g. `"pandas[pyarrow]"`).
            implementation: The [`Implementation`][] this constructor belongs to.
            requirements: Package names that must be importable for this constructor
                to be available (checked via `importlib.util.find_spec`).
            is_eager: Whether the backend returns an eager dataframe.
            is_nullable: Whether the backend has native null support.
            needs_gpu: Whether the backend requires GPU hardware.

        Returns:
            A decorator that replaces `func` with a `frame_constructor`
            instance registered into the shared `_registry`.
        """

        def decorator(func: Callable[Concatenate[Data, ...], R]) -> frame_constructor[R]:
            inst: frame_constructor[R] = frame_constructor(
                func,
                name=name,
                implementation=implementation,
                requirements=requirements,
                is_eager=is_eager,
                is_nullable=is_nullable,
                needs_gpu=needs_gpu,
            )
            cls._registry[name] = inst
            return inst

        return decorator

    @overload
    def __call__(
        self: frame_constructor[IntoDataFrameT],
        obj: Data,
        /,
        namespace: ModuleType,
        **kwds: Any,
    ) -> DataFrame[IntoDataFrameT]: ...
    @overload
    def __call__(
        self: frame_constructor[IntoLazyFrameT],
        obj: Data,
        /,
        namespace: ModuleType,
        **kwds: Any,
    ) -> LazyFrame[IntoLazyFrameT]: ...
    @overload
    def __call__(
        self: frame_constructor[IntoFrame],
        obj: Data,
        /,
        namespace: ModuleType,
        **kwds: Any,
    ) -> DataFrame[Any] | LazyFrame[Any]: ...

    def __call__(
        self, obj: Data, /, namespace: ModuleType, **kwds: Any
    ) -> DataFrame[Any] | LazyFrame[Any]:
        """Build a native frame and wrap it with `namespace.from_native`.

        Arguments:
            obj: Column-oriented mapping passed to the wrapped builder.
            namespace: A narwhals namespace (e.g. `narwhals`, `narwhals.stable.v1`)
                whose `from_native` performs the wrapping.
            **kwds: Forwarded to the wrapped builder.
        """
        native = self.func(obj, **kwds)
        return namespace.from_native(native)  # type: ignore[no-any-return]

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
        # TODO(FBruzzesi): Remove once all the `"backend" in str(constructor)`
        # statements in the test suite are properly replaced
        return self.func.__name__

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

    def __hash__(self) -> int:
        return hash((type(self), self.name))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, frame_constructor) and self.name == other.name


# Eager constructors


@frame_constructor.register(
    name="pandas",
    implementation=Implementation.PANDAS,
    requirements=("pandas",),
    is_eager=True,
    is_nullable=False,
)
def pandas_constructor(obj: Data, /, **kwds: Any) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(obj, **kwds)


@frame_constructor.register(
    name="pandas[nullable]",
    implementation=Implementation.PANDAS,
    requirements=("pandas",),
    is_eager=True,
)
def pandas_nullable_constructor(obj: Data, /, **kwds: Any) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(obj, **kwds).convert_dtypes(dtype_backend="numpy_nullable")


@frame_constructor.register(
    name="pandas[pyarrow]",
    implementation=Implementation.PANDAS,
    requirements=("pandas", "pyarrow"),
    is_eager=True,
)
def pandas_pyarrow_constructor(obj: Data, /, **kwds: Any) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(obj, **kwds).convert_dtypes(dtype_backend="pyarrow")


@frame_constructor.register(
    name="pyarrow",
    implementation=Implementation.PYARROW,
    requirements=("pyarrow",),
    is_eager=True,
)
def pyarrow_table_constructor(obj: Data, /, **kwds: Any) -> pa.Table:
    import pyarrow as pa

    return pa.table(obj, **kwds)


@frame_constructor.register(
    name="modin",
    implementation=Implementation.MODIN,
    requirements=("modin",),
    is_eager=True,
    is_nullable=False,
)
def modin_constructor(obj: Data, /, **kwds: Any) -> IntoDataFrame:  # pragma: no cover
    import modin.pandas as mpd
    import pandas as pd

    return cast("IntoDataFrame", mpd.DataFrame(pd.DataFrame(obj, **kwds)))


@frame_constructor.register(
    name="modin[pyarrow]",
    implementation=Implementation.MODIN,
    requirements=("modin", "pyarrow"),
    is_eager=True,
)
def modin_pyarrow_constructor(
    obj: Data, /, **kwds: Any
) -> IntoDataFrame:  # pragma: no cover
    import modin.pandas as mpd
    import pandas as pd

    df = mpd.DataFrame(pd.DataFrame(obj, **kwds)).convert_dtypes(dtype_backend="pyarrow")
    return cast("IntoDataFrame", df)


@frame_constructor.register(
    name="cudf",
    implementation=Implementation.CUDF,
    requirements=("cudf",),
    is_eager=True,
    needs_gpu=True,
)
def cudf_constructor(obj: Data, /, **kwds: Any) -> IntoDataFrame:  # pragma: no cover
    import cudf

    return cast("IntoDataFrame", cudf.DataFrame(obj, **kwds))


@frame_constructor.register(
    name="polars[eager]",
    implementation=Implementation.POLARS,
    requirements=("polars",),
    is_eager=True,
)
def polars_eager_constructor(obj: Data, /, **kwds: Any) -> pl.DataFrame:
    import polars as pl

    return pl.DataFrame(obj, **kwds)


# Lazy constructors


@frame_constructor.register(
    name="polars[lazy]", implementation=Implementation.POLARS, requirements=("polars",)
)
def polars_lazy_constructor(obj: Data, /, **kwds: Any) -> pl.LazyFrame:
    import polars as pl

    return pl.LazyFrame(obj, **kwds)


@frame_constructor.register(
    name="dask",
    implementation=Implementation.DASK,
    requirements=("dask",),
    is_nullable=False,
)
def dask_lazy_p2_constructor(
    obj: Data, /, npartitions: int = 2, **kwds: Any
) -> NativeDask:  # pragma: no cover
    import dask.dataframe as dd

    return cast("NativeDask", dd.from_dict(obj, npartitions=npartitions, **kwds))


@frame_constructor.register(
    name="duckdb",
    implementation=Implementation.DUCKDB,
    requirements=("duckdb", "pyarrow"),
)
def duckdb_lazy_constructor(obj: Data, /, **kwds: Any) -> NativeDuckDB:
    import duckdb
    import pyarrow as pa

    duckdb.sql("""set timezone = 'UTC'""")
    _df = pa.table(obj, **kwds)
    return duckdb.sql("select * from _df")


def _pyspark_build(obj: Data, /, **kwds: Any) -> NativePySpark:  # pragma: no cover
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


@frame_constructor.register(
    name="pyspark", implementation=Implementation.PYSPARK, requirements=("pyspark",)
)
def pyspark_lazy_constructor(
    obj: Data, /, **kwds: Any
) -> NativePySpark:  # pragma: no cover
    return _pyspark_build(obj, **kwds)


@frame_constructor.register(
    name="pyspark[connect]",
    implementation=Implementation.PYSPARK_CONNECT,
    requirements=("pyspark",),
)
def pyspark_connect_lazy_constructor(
    obj: Data, /, **kwds: Any
) -> NativePySpark:  # pragma: no cover
    return _pyspark_build(obj, **kwds)


@frame_constructor.register(
    name="sqlframe",
    implementation=Implementation.SQLFRAME,
    requirements=("sqlframe", "duckdb"),
)
def sqlframe_pyspark_lazy_constructor(obj: Data, /, **kwds: Any) -> NativeSQLFrame:
    session = sqlframe_session()
    return session.createDataFrame([*zip(*obj.values())], schema=[*obj.keys()], **kwds)


@frame_constructor.register(
    name="ibis",
    implementation=Implementation.IBIS,
    requirements=("ibis", "duckdb", "pyarrow"),
)
def ibis_lazy_constructor(obj: Data, /, **kwds: Any) -> ibis.Table:  # pragma: no cover
    import pyarrow as pa

    table = pa.table(obj)
    table_name = str(uuid.uuid4())
    return _ibis_backend().create_table(table_name, table, **kwds)


DEFAULT_BACKENDS: frozenset[str] = frozenset(
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
"""Subset of backends enabled by default for parametrised tests when the
user does not pass `--nw-backends` (mirrors the historical Narwhals defaults).
"""


def available_backends() -> frozenset[str]:
    """Return the names of every constructor whose backend is importable.

    Examples:
        >>> from narwhals.testing.constructors import available_backends
        >>> "pandas" in available_backends()
        True
    """
    return frozenset(
        name for name, c in frame_constructor._registry.items() if c.is_available
    )


def available_cpu_backends() -> frozenset[str]:  # pragma: no cover
    """Return the names of every CPU constructor whose backend is importable.

    Examples:
        >>> from narwhals.testing.constructors import available_cpu_backends
        >>> "pandas" in available_cpu_backends()
        True
    """
    return frozenset(
        name
        for name, c in frame_constructor._registry.items()
        if c.is_available and not c.needs_gpu
    )


EagerName: TypeAlias = Literal[
    "pandas",
    "pandas[nullable]",
    "pandas[pyarrow]",
    "modin",
    "modin[pyarrow]",
    "cudf",
    "polars[eager]",
    "pyarrow",
]
LazyName: TypeAlias = Literal[
    "polars[lazy]", "dask", "duckdb", "pyspark", "pyspark[connect]", "sqlframe", "ibis"
]


@overload
def get_backend_constructor(name: EagerName) -> frame_constructor[IntoDataFrame]: ...
@overload
def get_backend_constructor(name: LazyName) -> frame_constructor[IntoLazyFrame]: ...
@overload
def get_backend_constructor(name: str) -> frame_constructor[IntoFrame]: ...


def get_backend_constructor(name: str) -> frame_constructor[IntoFrame]:
    """Return the registered constructor for `name`.

    Arguments:
        name: The string identifier of a registered constructor
            (e.g. `"pandas[pyarrow]"`).

    Raises:
        ValueError: If `name` is not a registered constructor identifier.

    Examples:
        >>> from narwhals.testing.constructors import get_backend_constructor
        >>> get_backend_constructor("pandas")
        frame_constructor(name='pandas')
    """
    try:
        return frame_constructor._registry[name]
    except KeyError as exc:
        valid = sorted(frame_constructor._registry)
        msg = f"Unknown constructor {name!r}. Expected one of: {valid}."
        raise ValueError(msg) from exc


def prepare_backends(
    *, include: Iterable[str] | None = None, exclude: Iterable[str] | None = None
) -> list[frame_constructor[IntoFrame]]:
    """Return available constructors, optionally filtered.

    Note:
        `exclude` is given precedence in the selection.

    Arguments:
        include: If given, only return backends whose name is in this set.
        exclude: If given, remove backends whose name is in this set.

    Examples:
        >>> from narwhals.testing.constructors import prepare_backends
        >>> backends = prepare_backends(include=["pandas", "polars[eager]"])
    """
    available = available_backends()
    candidates: list[frame_constructor[Any]] = [
        c for name, c in frame_constructor._registry.items() if name in available
    ]

    include_set: frozenset[str] = (
        frozenset(include) if include is not None else frozenset()
    )
    exclude_set: frozenset[str] = (
        frozenset(exclude) if exclude is not None else frozenset()
    )

    if unknown := (include_set.union(exclude_set).difference(available)):
        msg = f"The following names are not known constructors: {sorted(unknown)}"
        raise ValueError(msg)

    if include is not None:
        candidates = [c for c in candidates if c.name in include_set]
    if exclude is not None:
        candidates = [c for c in candidates if c.name not in exclude_set]
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
