# pandas / Polars / etc. : if a user passes a dataframe from one of these
# libraries, it means they must already have imported the given module.
# So, we can just check sys.modules.
from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard
    import pandas as pd


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


def get_pyarrow_parquet() -> Any:  # pragma: no cover
    """Get pyarrow.parquet module (if pyarrow has already been imported - else return None)."""
    if "pyarrow" in sys.modules:
        import pyarrow.parquet as pp

        return pp
    return None


def get_numpy() -> Any:
    """Get numpy module (if already imported - else return None)."""
    return sys.modules.get("numpy", None)


def get_dask() -> Any:
    """Get dask (if already imported - else return None)."""
    return sys.modules.get("dask", None)


def get_dask_dataframe() -> Any:
    """Get dask.dataframe module (if already imported - else return None)."""
    return sys.modules.get("dask.dataframe", None)


def get_dask_expr() -> Any:
    """Get dask_expr module (if already imported - else return None)."""
    return sys.modules.get("dask_expr", None)


def is_pandas_dataframe(df: Any) -> TypeGuard[pd.DataFrame]:
    """Check whether `df` is a pandas DataFrame without importing pandas."""
    return bool((pd := get_pandas()) is not None and isinstance(df, pd.DataFrame))


__all__ = [
    "get_polars",
    "get_pandas",
    "get_modin",
    "get_cudf",
    "get_pyarrow",
    "get_pyarrow_compute",
    "get_numpy",
    "is_pandas_dataframe",
]
