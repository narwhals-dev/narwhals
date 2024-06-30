# pandas / Polars / etc. : if a user passes a dataframe from one of these
# libraries, it means they must already have imported the given module.
# So, we can just check sys.modules.
from __future__ import annotations

import sys
from enum import Enum
from enum import auto
from typing import TYPE_CHECKING
from typing import Any

from typing_extensions import assert_never

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard
    import pandas


class Implementation(Enum):
    POLARS = auto()
    PANDAS = auto()
    MODIN = auto()
    CUDF = auto()
    PYARROW = auto()
    NUMPY = auto()


def get_polars() -> Any:
    """Get Polars module (if already imported - else return None)."""
    return sys.modules.get("polars", None)


def get_pandas() -> Any:
    """Get pandas module (if already imported - else return None)."""
    return sys.modules.get("pandas", None)


def get_modin() -> Any:  # pragma: no cover
    """Get modin.pandas module (if already imported - else return None)."""
    if (modin := sys.modules.get("modin", None)) is not None:
        return modin.pandas
    return None


def get_cudf() -> Any:
    """Get cudf module (if already imported - else return None)."""
    return sys.modules.get("cudf", None)


def get_pyarrow() -> Any:  # pragma: no cover
    """Get pyarrow module (if already imported - else return None)."""
    return sys.modules.get("pyarrow", None)


def get_pyarrow_compute() -> Any:  # pragma: no cover
    """Get pyarrow.compute module (if pyarrow has already been imported - else return None)."""
    if "pyarrow" in sys.modules:
        import pyarrow.compute as pc

        return pc
    return None


def get_numpy() -> Any:
    """Get numpy module (if already imported - else return None)."""
    return sys.modules.get("numpy", None)


def get_backend(implementation: Implementation) -> Any:
    # This is equivalent to an exhaustive match without using Python 3.10 match
    # More info in
    # https://adamj.eu/tech/2022/10/14/python-type-hints-exhuastiveness-checking

    if implementation is Implementation.POLARS:  # pragma: no cover
        return get_polars()
    if implementation is Implementation.PANDAS:
        return get_pandas()
    if implementation is Implementation.MODIN:
        return get_modin()
    if implementation is Implementation.CUDF:
        return get_cudf()
    if implementation is Implementation.PYARROW:  # pragma: no cover
        return get_pyarrow()
    if implementation is Implementation.NUMPY:  # pragma: no cover
        return get_numpy()

    return assert_never(implementation)


def is_pandas_dataframe(df: Any) -> TypeGuard[pandas.DataFrame]:
    """Check whether `df` is a pandas DataFrame without importing pandas."""
    if (pd := get_pandas()) is not None and isinstance(df, pd.DataFrame):
        return True
    return False


__all__ = [
    "get_backend",
    "get_polars",
    "get_pandas",
    "get_modin",
    "get_cudf",
    "get_pyarrow",
    "get_pyarrow_compute",
    "get_numpy",
    "is_pandas_dataframe",
]
