# pandas / Polars / etc. : if a user passes a dataframe from one of these
# libraries, it means they must already have imported the given module.
# So, we can just check sys.modules.

import sys
from enum import Enum
from enum import auto
from typing import Any

from typing_extensions import assert_never


class Backend(Enum):
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


def get_implementation(backend: Backend) -> Any:
    # This is equivalent to an exhaustive match without using Python 3.10 match
    # More info in
    # https://adamj.eu/tech/2022/10/14/python-type-hints-exhuastiveness-checking

    if backend is Backend.POLARS:
        return get_polars()
    if backend is Backend.PANDAS:
        return get_pandas()
    if backend is Backend.MODIN:
        return get_modin()
    if backend is Backend.CUDF:
        return get_cudf()
    if backend is Backend.PYARROW:
        return get_pyarrow()
    if backend is Backend.NUMPY:
        return get_numpy()

    return assert_never(backend)
