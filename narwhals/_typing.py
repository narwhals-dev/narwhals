from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Literal, Union

from narwhals._typing_compat import TypeVar
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

# `str` aliases
_Polars: TypeAlias = Literal["polars"]
_Arrow: TypeAlias = Literal["pyarrow"]
_Dask: TypeAlias = Literal["dask"]
_DuckDB: TypeAlias = Literal["duckdb"]
_Pandas: TypeAlias = Literal["pandas"]
_Modin: TypeAlias = Literal["modin"]
_Cudf: TypeAlias = Literal["cudf"]
_PySpark: TypeAlias = Literal["pyspark"]
_SQLFrame: TypeAlias = Literal["sqlframe"]
_PySparkConnect: TypeAlias = Literal["pyspark[connect]"]
_Ibis: TypeAlias = Literal["ibis"]
_PandasLike: TypeAlias = Literal[_Pandas, _Cudf, _Modin]
_SparkLike: TypeAlias = Literal[_PySpark, _SQLFrame, _PySparkConnect]
_EagerOnly: TypeAlias = Literal[_PandasLike, _Arrow]
_EagerAllowed: TypeAlias = Literal[_Polars, _EagerOnly]
_LazyOnly: TypeAlias = Literal[_SparkLike, _Dask, _DuckDB, _Ibis]
_LazyAllowed: TypeAlias = Literal[_Polars, _LazyOnly]

# `Implementation` aliases
PANDAS: TypeAlias = Literal[Implementation.PANDAS]
MODIN: TypeAlias = Literal[Implementation.MODIN]
CUDF: TypeAlias = Literal[Implementation.CUDF]
PYSPARK: TypeAlias = Literal[Implementation.PYSPARK]
SQLFRAME: TypeAlias = Literal[Implementation.SQLFRAME]
PYSPARK_CONNECT: TypeAlias = Literal[Implementation.PYSPARK_CONNECT]
POLARS: TypeAlias = Literal[Implementation.POLARS]
ARROW: TypeAlias = Literal[Implementation.PYARROW]
DASK: TypeAlias = Literal[Implementation.DASK]
DUCKDB: TypeAlias = Literal[Implementation.DUCKDB]
IBIS: TypeAlias = Literal[Implementation.IBIS]
PANDAS_LIKE: TypeAlias = Literal[PANDAS, CUDF, MODIN]
SPARK_LIKE: TypeAlias = Literal[PYSPARK, SQLFRAME, PYSPARK_CONNECT]
EAGER_ONLY: TypeAlias = Literal[PANDAS_LIKE, ARROW]
EAGER_ALLOWED: TypeAlias = Literal[EAGER_ONLY, POLARS]
LAZY_ONLY: TypeAlias = Literal[SPARK_LIKE, DASK, DUCKDB, IBIS]
LAZY_ALLOWED: TypeAlias = Literal[LAZY_ONLY, POLARS]

# `str | Implementation` aliases
Pandas: TypeAlias = Literal[_Pandas, PANDAS]
Cudf: TypeAlias = Literal[_Cudf, CUDF]
Modin: TypeAlias = Literal[_Modin, MODIN]
PySpark: TypeAlias = Literal[_PySpark, PYSPARK]
SQLFrame: TypeAlias = Literal[_SQLFrame, SQLFRAME]
PySparkConnect: TypeAlias = Literal[_PySparkConnect, PYSPARK_CONNECT]
Polars: TypeAlias = Literal[_Polars, POLARS]
"""Maybe we could say something here?"""

Arrow: TypeAlias = Literal[_Arrow, ARROW]
Dask: TypeAlias = Literal[_Dask, DASK]
"""Might want to highlight the limited support?

After (#2858), link to https://narwhals-dev.github.io/narwhals/api-completeness/expr/
"""

DuckDB: TypeAlias = Literal[_DuckDB, DUCKDB]
Ibis: TypeAlias = Literal[_Ibis, IBIS]
PandasLike: TypeAlias = Literal[_PandasLike, PANDAS_LIKE]
SparkLike: TypeAlias = Literal[_SparkLike, SPARK_LIKE]
EagerOnly: TypeAlias = Literal[PandasLike, Arrow]
EagerAllowed: TypeAlias = Literal[EagerOnly, Polars]
LazyOnly: TypeAlias = Literal[SparkLike, Dask, DuckDB, Ibis]
LazyAllowed: TypeAlias = Literal[LazyOnly, Polars]

BackendName: TypeAlias = Literal[_EagerAllowed, _LazyAllowed]
Backend: TypeAlias = Literal[EagerAllowed, LazyAllowed]
"""Ooh look, a description!"""


BackendT = TypeVar("BackendT", bound=Backend)
IntoBackend: TypeAlias = Union[BackendT, ModuleType]
"""Here's lots of long text.

Even more stuff here.

And here too!

Examples:

    # not a doctest!
    import narwhals as nw

    nw.hello()
"""


IntoBackendAny: TypeAlias = IntoBackend[Backend]
IntoBackendEager: TypeAlias = IntoBackend[EagerAllowed]
IntoBackendLazy: TypeAlias = IntoBackend[LazyAllowed]


def dataframe_lazy_current(backend: IntoBackendLazy | None = None) -> None: ...
def dataframe_lazy_alt_1(
    backend: IntoBackend[Polars | DuckDB | Ibis | Dask] | None = None,
) -> None: ...


def lazyframe_collect_current(backend: IntoBackendEager | None = None) -> None: ...
def lazyframe_collect_alt_1(
    backend: IntoBackend[Polars | Pandas | Arrow] | None = None,
) -> None: ...
