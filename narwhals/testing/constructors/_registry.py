"""Registry of constructors that ship with `narwhals.testing`."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from narwhals.testing.constructors._classes import (
    ConstructorBase,
    CudfConstructor,
    DaskConstructor,
    DuckDBConstructor,
    IbisConstructor,
    ModinConstructor,
    ModinPyArrowConstructor,
    PandasConstructor,
    PandasNullableConstructor,
    PandasPyArrowConstructor,
    PolarsEagerConstructor,
    PolarsLazyConstructor,
    PyArrowConstructor,
    PySparkConnectConstructor,
    PySparkConstructor,
    SQLFrameConstructor,
)
from narwhals.testing.constructors._name import ConstructorName

if TYPE_CHECKING:
    from collections.abc import Iterable


# Singleton instance per backend. Users that need a non-default parametrisation
# (e.g. `DaskConstructor(npartitions=1)`) can instantiate the class directly.
ALL_CONSTRUCTORS: dict[ConstructorName, ConstructorBase] = {
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

DEFAULT_CONSTRUCTORS: frozenset[ConstructorName] = frozenset(
    {
        ConstructorName.PANDAS,
        ConstructorName.PANDAS_PYARROW,
        ConstructorName.POLARS_EAGER,
        ConstructorName.PYARROW,
        ConstructorName.DUCKDB,
        ConstructorName.SQLFRAME,
        ConstructorName.IBIS,
    }
)
"""Subset of constructors enabled by default for parametrised tests when the
user does not pass `--constructors` (mirrors the historical Narwhals defaults).
"""

# All constructors that don't require a GPU. Useful for `--all-cpu-constructors`.
ALL_CPU_CONSTRUCTORS: frozenset[ConstructorName] = frozenset(
    name for name in ALL_CONSTRUCTORS if not name.needs_gpu
)


# Map from `ConstructorName` to the package import name that needs to be
# importable for that constructor to work. Some backends have extra
# requirements (e.g. `pandas[pyarrow]` also needs `pyarrow` installed); we
# encode those as tuples here so `is_backend_available` can check all of them.
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


def is_backend_available(name: ConstructorName) -> bool:
    """Whether all backends required by `name` can be imported in this environment.

    Examples:
        >>> from narwhals.testing.constructors import (
        ...     ConstructorName,
        ...     is_backend_available,
        ... )
        >>> is_backend_available(ConstructorName.PANDAS)
        True
    """
    return all(find_spec(pkg) is not None for pkg in _BACKEND_REQUIREMENTS[name])


def available_constructors() -> frozenset[ConstructorName]:
    """Return every [`ConstructorName`][] whose backend is importable.

    Examples:
        >>> from narwhals.testing.constructors import available_constructors
        >>> ConstructorName.PANDAS in available_constructors()
        True
    """
    return frozenset(name for name in ALL_CONSTRUCTORS if is_backend_available(name))


def get_constructor(name: ConstructorName | str) -> ConstructorBase:
    """Return the registered singleton constructor for `name`.

    Arguments:
        name: A [`ConstructorName`][] member or its string value
            (e.g. `"pandas[pyarrow]"`).

    Raises:
        ValueError: If `name` is not a registered constructor identifier.

    Examples:
        >>> from narwhals.testing.constructors import get_constructor
        >>> get_constructor("pandas")
        PandasConstructor()
    """
    try:
        key = ConstructorName(name) if isinstance(name, str) else name
    except ValueError as exc:
        valid = sorted(c.value for c in ConstructorName)
        msg = f"Unknown constructor {name!r}. Expected one of: {valid}."
        raise ValueError(msg) from exc
    return ALL_CONSTRUCTORS[key]


def resolve_constructors(names: Iterable[ConstructorName | str]) -> list[ConstructorBase]:
    """Resolve an iterable of names / identifiers into a list of constructor instances.

    Order is preserved; duplicates are kept (so the same constructor can be
    parametrised multiple times if explicitly requested).
    """
    return [get_constructor(n) for n in names]


__all__ = [
    "ALL_CONSTRUCTORS",
    "ALL_CPU_CONSTRUCTORS",
    "DEFAULT_CONSTRUCTORS",
    "available_constructors",
    "get_constructor",
    "is_backend_available",
    "resolve_constructors",
]
