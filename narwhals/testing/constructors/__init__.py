from __future__ import annotations

from narwhals.testing.constructors._classes import (
    ConstructorBase,
    ConstructorEagerBase,
    pyspark_session,
    sqlframe_session,
)
from narwhals.testing.constructors._name import ConstructorName

__all__ = (
    "ALL_CONSTRUCTORS",
    "ALL_CPU_CONSTRUCTORS",
    "DEFAULT_CONSTRUCTORS",
    "ConstructorBase",
    "ConstructorEagerBase",
    "ConstructorName",
    "available_constructors",
    "get_constructor",
    "pyspark_session",
    "sqlframe_session",
)


ALL_CONSTRUCTORS: dict[ConstructorName, ConstructorBase] = {
    name: name.constructor for name in ConstructorName
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

ALL_CPU_CONSTRUCTORS: frozenset[ConstructorName] = frozenset(
    name for name in ConstructorName if not name.needs_gpu
)


def available_constructors() -> frozenset[ConstructorName]:
    """Return every [`ConstructorName`][] whose backend is importable.

    Examples:
        >>> from narwhals.testing.constructors import available_constructors
        >>> ConstructorName.PANDAS in available_constructors()
        True
    """
    return frozenset(name for name in ConstructorName if name.is_available)


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
