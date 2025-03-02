from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Mapping
from typing import Sequence

import pandas as pd
import pyarrow as pa

import narwhals as nw
from narwhals.translate import from_native
from narwhals.typing import IntoDataFrame
from narwhals.typing import IntoFrame
from narwhals.utils import Implementation
from narwhals.utils import parse_version

if sys.version_info >= (3, 10):
    from typing import TypeAlias  # pragma: no cover
else:
    from typing_extensions import TypeAlias  # pragma: no cover


def get_module_version_as_tuple(module_name: str) -> tuple[int, ...]:
    try:
        return parse_version(__import__(module_name).__version__)
    except ImportError:
        return (0, 0, 0)


IBIS_VERSION: tuple[int, ...] = get_module_version_as_tuple("ibis")
NUMPY_VERSION: tuple[int, ...] = get_module_version_as_tuple("numpy")
PANDAS_VERSION: tuple[int, ...] = get_module_version_as_tuple("pandas")
DUCKDB_VERSION: tuple[int, ...] = get_module_version_as_tuple("duckdb")
POLARS_VERSION: tuple[int, ...] = get_module_version_as_tuple("polars")
DASK_VERSION: tuple[int, ...] = get_module_version_as_tuple("dask")
PYARROW_VERSION: tuple[int, ...] = get_module_version_as_tuple("pyarrow")
PYSPARK_VERSION: tuple[int, ...] = get_module_version_as_tuple("pyspark")

Constructor: TypeAlias = Callable[[Any], IntoFrame]
ConstructorEager: TypeAlias = Callable[[Any], IntoDataFrame]


def zip_strict(left: Sequence[Any], right: Sequence[Any]) -> Iterator[Any]:
    if len(left) != len(right):
        msg = f"left {len(left)=} != right {len(right)=}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    return zip(left, right)


def _to_comparable_list(column_values: Any) -> Any:
    if isinstance(column_values, nw.Series) and isinstance(
        column_values.to_native(), pa.Array
    ):  # pragma: no cover
        # Narwhals Series for PyArrow should be backed by ChunkedArray, not Array.
        msg = "Did not expect to see Arrow Array here"
        raise TypeError(msg)
    if (
        hasattr(column_values, "_compliant_series")
        and column_values._compliant_series._implementation is Implementation.CUDF
    ):  # pragma: no cover
        column_values = column_values.to_pandas()
    if hasattr(column_values, "to_list"):
        return column_values.to_list()
    return list(column_values)


def _sort_dict_by_key(
    data_dict: Mapping[str, list[Any]], key: str
) -> dict[str, list[Any]]:  # pragma: no cover
    sort_list = data_dict[key]
    sorted_indices = sorted(
        range(len(sort_list)),
        key=lambda i: (
            (sort_list[i] is None)
            or (isinstance(sort_list[i], float) and math.isnan(sort_list[i])),
            sort_list[i],
        ),
    )
    return {key: [value[i] for i in sorted_indices] for key, value in data_dict.items()}


def assert_equal_data(result: Any, expected: Mapping[str, Any]) -> None:
    is_pyspark = (
        hasattr(result, "_compliant_frame")
        and result.implementation is Implementation.PYSPARK
    )
    is_duckdb = (
        hasattr(result, "_compliant_frame")
        and result._compliant_frame._implementation is Implementation.DUCKDB
    )
    if is_duckdb:
        result = from_native(result.to_native().arrow())
    if hasattr(result, "collect"):
        kwargs: dict[Implementation, dict[str, Any]] = {Implementation.POLARS: {}}

        if os.environ.get("NARWHALS_POLARS_GPU", None):  # pragma: no cover
            kwargs[Implementation.POLARS].update({"engine": "gpu"})
        if os.environ.get("NARWHALS_POLARS_NEW_STREAMING", None):  # pragma: no cover
            kwargs[Implementation.POLARS].update({"new_streaming": True})

        result = result.collect(**kwargs.get(result.implementation, {}))

    if hasattr(result, "columns"):
        for idx, (col, key) in enumerate(zip(result.columns, expected.keys())):
            assert col == key, f"Expected column name {key} at index {idx}, found {col}"
    result = {key: _to_comparable_list(result[key]) for key in expected}
    if (is_pyspark or is_duckdb) and expected:  # pragma: no cover
        sort_key = next(iter(expected.keys()))
        expected = _sort_dict_by_key(expected, sort_key)
        result = _sort_dict_by_key(result, sort_key)
    assert list(result.keys()) == list(expected.keys()), (
        f"Result keys {result.keys()}, expected keys: {expected.keys()}"
    )

    for key, expected_value in expected.items():
        result_value = result[key]
        for i, (lhs, rhs) in enumerate(zip_strict(result_value, expected_value)):
            if isinstance(lhs, float) and not math.isnan(lhs):
                are_equivalent_values = rhs is not None and math.isclose(
                    lhs, rhs, rel_tol=0, abs_tol=1e-6
                )
            elif isinstance(lhs, float) and math.isnan(lhs):
                are_equivalent_values = rhs is None or math.isnan(rhs)
            elif isinstance(rhs, float) and math.isnan(rhs):
                are_equivalent_values = lhs is None or math.isnan(lhs)
            elif lhs is None:
                are_equivalent_values = rhs is None
            elif isinstance(lhs, list) and isinstance(rhs, list):
                are_equivalent_values = all(
                    left_side == right_side for left_side, right_side in zip(lhs, rhs)
                )
            elif pd.isna(lhs):
                are_equivalent_values = pd.isna(rhs)
            else:
                are_equivalent_values = lhs == rhs
            assert are_equivalent_values, (
                f"Mismatch at index {i}: {lhs} != {rhs}\nExpected: {expected}\nGot: {result}"
            )


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


def is_windows() -> bool:
    """Check if the current platform is Windows."""
    return sys.platform in {"win32", "cygwin"}


def windows_has_tzdata() -> bool:  # pragma: no cover
    """From PyArrow: python/pyarrow/tests/util.py."""
    return (Path.home() / "Downloads" / "tzdata").exists()


def is_pyarrow_windows_no_tzdata(constructor: Constructor, /) -> bool:
    """Skip test on Windows when the tz database is not configured."""
    return "pyarrow" in str(constructor) and is_windows() and not windows_has_tzdata()
