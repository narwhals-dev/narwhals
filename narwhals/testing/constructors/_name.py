from __future__ import annotations

from enum import Enum
from importlib.util import find_spec
from typing import TYPE_CHECKING

from narwhals._utils import Implementation

if TYPE_CHECKING:
    import pytest

    from narwhals.testing.constructors._classes import ConstructorBase


def is_backend_available(*packages: str) -> bool:
    """Whether all backends required by `name` can be imported in this environment.

    Examples:
        >>> from narwhals.testing.constructors import (
        ...     ConstructorName,
        ...     is_backend_available,
        ... )
        >>> is_backend_available(ConstructorName.PANDAS)
        True
    """
    return all(find_spec(pkg) is not None for pkg in packages)


class ConstructorName(str, Enum):
    """Typed identifier for each backend exposed by `narwhals.testing.constructors`.

    The string values are byte-identical to the identifiers accepted by the
    `--constructors` pytest CLI option (e.g. `pandas[pyarrow]`, `polars[lazy]`).

    Examples:
        >>> from narwhals.testing.constructors import ConstructorName
        >>> ConstructorName.PANDAS_PYARROW.value
        'pandas[pyarrow]'
        >>> ConstructorName.PANDAS_PYARROW.is_pandas_like
        True
        >>> ConstructorName.PANDAS_PYARROW.needs_pyarrow
        True
    """

    PANDAS = "pandas"
    PANDAS_NULLABLE = "pandas[nullable]"
    PANDAS_PYARROW = "pandas[pyarrow]"
    PYARROW = "pyarrow"
    MODIN = "modin"
    MODIN_PYARROW = "modin[pyarrow]"
    CUDF = "cudf"
    POLARS_EAGER = "polars[eager]"
    POLARS_LAZY = "polars[lazy]"
    DASK = "dask"
    DUCKDB = "duckdb"
    PYSPARK = "pyspark"
    PYSPARK_CONNECT = "pyspark[connect]"
    SQLFRAME = "sqlframe"
    IBIS = "ibis"

    def __str__(self) -> str:
        return str(self.value)

    @property
    def implementation(self) -> Implementation:
        """The [`Implementation`][] that this constructor belongs to."""
        return _NAME_TO_IMPL[self]

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
    def is_eager(self) -> bool:
        """Whether this constructor produces an eager native dataframe."""
        return self in {
            ConstructorName.PANDAS,
            ConstructorName.PANDAS_NULLABLE,
            ConstructorName.PANDAS_PYARROW,
            ConstructorName.PYARROW,
            ConstructorName.MODIN,
            ConstructorName.MODIN_PYARROW,
            ConstructorName.CUDF,
            ConstructorName.POLARS_EAGER,
        }

    @property
    def is_lazy(self) -> bool:
        """Whether this constructor produces a lazy native frame."""
        return not self.is_eager

    @property
    def needs_pyarrow(self) -> bool:
        """Whether this constructor requires `pyarrow` to be installed."""
        return self in {
            ConstructorName.PYARROW,
            ConstructorName.PANDAS_PYARROW,
            ConstructorName.MODIN_PYARROW,
            ConstructorName.DUCKDB,
            ConstructorName.IBIS,
        }

    @property
    def is_non_nullable(self) -> bool:
        """Whether this constructor uses a backend without native null support."""
        return self in {
            ConstructorName.PANDAS,
            ConstructorName.MODIN,
            ConstructorName.DASK,
        }

    @property
    def needs_gpu(self) -> bool:
        """Whether this constructor requires GPU hardware."""
        return self is ConstructorName.CUDF

    @classmethod
    def from_pytest_request(cls, request: pytest.FixtureRequest) -> ConstructorName:
        """Resolve the [`ConstructorName`][] from the current parametrised pytest request.

        Examples:
            >>> import pytest
            >>> def test_example(constructor, request):  # doctest: +SKIP
            ...     name = ConstructorName.from_pytest_request(request)
            ...     if name.is_pandas_like:
            ...         ...
        """
        return cls(str(request.node.callspec.id))

    @property
    def constructor(self) -> ConstructorBase:
        """Return the registered singleton constructor for this name."""
        from narwhals.testing.constructors._classes import ConstructorBase

        return ConstructorBase._registry[self]

    @property
    def is_available(self) -> bool:
        """Whether every package required by this constructor is importable."""
        from narwhals.testing.constructors._classes import ConstructorBase

        return is_backend_available(*ConstructorBase._requirements[self])


_NAME_TO_IMPL: dict[ConstructorName, Implementation] = {
    ConstructorName.PANDAS: Implementation.PANDAS,
    ConstructorName.PANDAS_NULLABLE: Implementation.PANDAS,
    ConstructorName.PANDAS_PYARROW: Implementation.PANDAS,
    ConstructorName.PYARROW: Implementation.PYARROW,
    ConstructorName.MODIN: Implementation.MODIN,
    ConstructorName.MODIN_PYARROW: Implementation.MODIN,
    ConstructorName.CUDF: Implementation.CUDF,
    ConstructorName.POLARS_EAGER: Implementation.POLARS,
    ConstructorName.POLARS_LAZY: Implementation.POLARS,
    ConstructorName.DASK: Implementation.DASK,
    ConstructorName.DUCKDB: Implementation.DUCKDB,
    ConstructorName.PYSPARK: Implementation.PYSPARK,
    ConstructorName.PYSPARK_CONNECT: Implementation.PYSPARK_CONNECT,
    ConstructorName.SQLFRAME: Implementation.SQLFRAME,
    ConstructorName.IBIS: Implementation.IBIS,
}
