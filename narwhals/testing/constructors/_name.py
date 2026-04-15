from __future__ import annotations

from enum import Enum
from importlib.util import find_spec
from typing import TYPE_CHECKING

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
    def is_pandas(self) -> bool:
        """Whether this is one of the pandas constructors."""
        return self in {
            ConstructorName.PANDAS,
            ConstructorName.PANDAS_NULLABLE,
            ConstructorName.PANDAS_PYARROW,
        }

    @property
    def is_modin(self) -> bool:
        """Whether this is one of the modin constructors."""
        return self in {ConstructorName.MODIN, ConstructorName.MODIN_PYARROW}

    @property
    def is_cudf(self) -> bool:
        """Whether this is the cudf constructor."""
        return self is ConstructorName.CUDF

    @property
    def is_pandas_like(self) -> bool:
        """Whether this constructor produces a pandas-like dataframe (pandas, modin, cudf)."""
        return self.is_pandas or self.is_modin or self.is_cudf

    @property
    def is_polars(self) -> bool:
        """Whether this is one of the polars constructors."""
        return self in {ConstructorName.POLARS_EAGER, ConstructorName.POLARS_LAZY}

    @property
    def is_pyarrow(self) -> bool:
        """Whether this is the pyarrow table constructor."""
        return self is ConstructorName.PYARROW

    @property
    def is_dask(self) -> bool:
        """Whether this is the dask constructor."""
        return self is ConstructorName.DASK

    @property
    def is_duckdb(self) -> bool:
        """Whether this is the duckdb constructor."""
        return self is ConstructorName.DUCKDB

    @property
    def is_pyspark(self) -> bool:
        """Whether this is one of the pyspark constructors."""
        return self in {ConstructorName.PYSPARK, ConstructorName.PYSPARK_CONNECT}

    @property
    def is_sqlframe(self) -> bool:
        """Whether this is the sqlframe constructor."""
        return self is ConstructorName.SQLFRAME

    @property
    def is_ibis(self) -> bool:
        """Whether this is the ibis constructor."""
        return self is ConstructorName.IBIS

    @property
    def is_spark_like(self) -> bool:
        """Whether this constructor uses a spark-like backend (pyspark, sqlframe)."""
        return self.is_pyspark or self.is_sqlframe

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
        from narwhals.testing.constructors._classes import _ALL_CONSTRUCTORS

        return _ALL_CONSTRUCTORS[self]

    @property
    def is_available(self) -> bool:
        from narwhals.testing.constructors._classes import _BACKEND_REQUIREMENTS

        return is_backend_available(*_BACKEND_REQUIREMENTS[self])


__all__ = ["ConstructorName"]
