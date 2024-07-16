from __future__ import annotations

import math
import sys
import warnings
from typing import Any
from typing import Iterator
from typing import Sequence

import pandas as pd


def zip_strict(left: Sequence[Any], right: Sequence[Any]) -> Iterator[Any]:
    if len(left) != len(right):
        msg = f"left {len(left)=} != right {len(right)=}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    return zip(left, right)


def compare_dicts(result: Any, expected: dict[str, Any]) -> None:
    if hasattr(result, "collect"):
        result = result.collect()
    if (
        hasattr(result, "_dataframe")
        and hasattr(result._dataframe, "_implementation")
        and result._dataframe._implementation == "dask"
    ):
        result = result.to_pandas()
    if hasattr(result, "columns"):
        for key in result.columns:
            assert key in expected
    for key in expected:
        for lhs, rhs in zip_strict(result[key], expected[key]):
            if hasattr(lhs, "as_py"):
                lhs = lhs.as_py()  # noqa: PLW2901
            if hasattr(rhs, "as_py"):  # pragma: no cover
                rhs = rhs.as_py()  # noqa: PLW2901
            if isinstance(lhs, float) and not math.isnan(lhs):
                assert math.isclose(lhs, rhs, rel_tol=0, abs_tol=1e-6), (lhs, rhs)
            elif isinstance(lhs, float) and math.isnan(lhs):
                assert math.isnan(rhs), (lhs, rhs)  # pragma: no cover
            elif pd.isna(lhs):
                assert pd.isna(rhs), (lhs, rhs)
            else:
                assert lhs == rhs, (lhs, rhs)


def maybe_get_modin_df(df_pandas: pd.DataFrame) -> Any:
    """Convert a pandas DataFrame to a Modin DataFrame if Modin is available."""
    try:
        import modin.pandas as mpd
    except ImportError:  # pragma: no cover
        return df_pandas.copy()
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return mpd.DataFrame(df_pandas.to_dict(orient="list"))


def maybe_get_dask_df(df_pandas: pd.DataFrame) -> Any:
    """Convert a pandas DataFrame to a Dask Dataframe if Dask is available."""
    try:
        import dask.dataframe as dd

    except ImportError:
        return df_pandas.copy()
    else:
        return dd.from_pandas(df_pandas, npartitions=1)


def is_windows() -> bool:
    """Check if the current platform is Windows."""
    return sys.platform in ["win32", "cygwin"]
