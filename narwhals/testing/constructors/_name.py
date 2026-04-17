from __future__ import annotations

from enum import Enum
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

    from narwhals._utils import Implementation
    from narwhals.testing.constructors._classes import ConstructorBase


def is_backend_available(*packages: str) -> bool:
    """Whether every package in `packages` can be imported in this environment.

    Examples:
        >>> from narwhals.testing.constructors._name import is_backend_available
        >>> is_backend_available("pandas")
        True
    """
    return all(find_spec(pkg) is not None for pkg in packages)


class ConstructorName(str, Enum):
    """Typed identifier for each backend exposed by `narwhals.testing.constructors`.

    The string values are byte-identical to the identifiers accepted by the
    `--constructors` pytest CLI option (e.g. `pandas[pyarrow]`, `polars[lazy]`).

    All static metadata (implementation, requirements, eager/lazy, nullability,
    GPU need) lives on the registered constructor class in `_classes.py`.
    Properties on this enum delegate to that class so there is one source of
    truth for each backend.

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
    def constructor(self) -> ConstructorBase:
        """Return the registered singleton constructor for this name."""
        from narwhals.testing.constructors._classes import ConstructorBase

        return ConstructorBase._registry[self]

    @property
    def implementation(self) -> Implementation:
        """The [`Implementation`][] that this constructor belongs to."""
        return self.constructor.implementation

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
        return self.constructor.is_eager

    @property
    def is_lazy(self) -> bool:
        """Whether this constructor produces a lazy native frame."""
        return self.constructor.is_lazy

    @property
    def needs_pyarrow(self) -> bool:
        """Whether this constructor requires `pyarrow` to be installed."""
        return self.constructor.needs_pyarrow

    @property
    def is_non_nullable(self) -> bool:
        """Whether this constructor uses a backend without native null support."""
        return self.constructor.is_non_nullable

    @property
    def needs_gpu(self) -> bool:
        """Whether this constructor requires GPU hardware."""
        return self.constructor.needs_gpu

    @property
    def is_available(self) -> bool:
        """Whether every package required by this constructor is importable."""
        return self.constructor.is_available

    # TODO(Unassigned): remove 'no cover' flag once used in test suite
    @classmethod
    def from_pytest_request(
        cls, request: pytest.FixtureRequest
    ) -> ConstructorName:  # pragma: no cover
        """Resolve the [`ConstructorName`][] from the current parametrised pytest request.

        Examples:
            >>> import pytest
            >>> def test_example(constructor, request):  # doctest: +SKIP
            ...     name = ConstructorName.from_pytest_request(request)
            ...     if name.is_pandas_like:
            ...         ...
        """
        return cls(str(request.node.callspec.id))
