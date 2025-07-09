from __future__ import annotations

import math
import os
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
import pyarrow as pa

import narwhals as nw
from narwhals._utils import Implementation, parse_version
from narwhals.translate import from_native

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from typing_extensions import TypeAlias

    from narwhals.typing import DataFrameLike, NativeFrame, NativeLazyFrame


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
CUDF_VERSION: tuple[int, ...] = get_module_version_as_tuple("cudf")

Constructor: TypeAlias = Callable[[Any], "NativeLazyFrame | NativeFrame | DataFrameLike"]
ConstructorEager: TypeAlias = Callable[[Any], "NativeFrame | DataFrameLike"]


def zip_strict(left: Sequence[Any], right: Sequence[Any]) -> Iterator[Any]:
    if len(left) != len(right):
        msg = f"{len(left)=} != {len(right)=}\nLeft: {left}\nRight: {right}"  # pragma: no cover
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


def assert_equal_data(result: Any, expected: Mapping[str, Any]) -> None:
    is_duckdb = (
        hasattr(result, "_compliant_frame")
        and result._compliant_frame._implementation is Implementation.DUCKDB
    )
    is_ibis = (
        hasattr(result, "_compliant_frame")
        and result._compliant_frame._implementation is Implementation.IBIS
    )
    is_spark_like = (
        hasattr(result, "_compliant_frame")
        and result._compliant_frame._implementation.is_spark_like()
    )
    if is_duckdb:
        result = from_native(result.to_native().arrow())
    if is_ibis:
        result = from_native(result.to_native().to_pyarrow())
    if hasattr(result, "collect"):
        kwargs: dict[Implementation, dict[str, Any]] = {Implementation.POLARS: {}}

        if os.environ.get("NARWHALS_POLARS_GPU", None):  # pragma: no cover
            kwargs[Implementation.POLARS].update({"engine": "gpu"})
        if os.environ.get("NARWHALS_POLARS_NEW_STREAMING", None):  # pragma: no cover
            kwargs[Implementation.POLARS].update({"new_streaming": True})

        result = result.collect(**kwargs.get(result.implementation, {}))

    if hasattr(result, "columns"):
        for idx, (col, key) in enumerate(
            zip_strict(result.columns, list(expected.keys()))
        ):
            assert col == key, f"Expected column name {key} at index {idx}, found {col}"
    result = {key: _to_comparable_list(result[key]) for key in expected}
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
            elif type(lhs) is date and type(rhs) is datetime:
                are_equivalent_values = datetime(lhs.year, lhs.month, lhs.day) == rhs
            elif (
                is_spark_like
                and isinstance(lhs, datetime)
                and isinstance(rhs, datetime)
                and rhs.tzinfo is None
                and lhs.tzinfo
            ):
                # PySpark converts timezone-naive to timezone-aware by default in many cases.
                # For now, we just assert that the local result matches the expected one.
                # https://github.com/narwhals-dev/narwhals/issues/2793
                are_equivalent_values = lhs.replace(tzinfo=None) == rhs
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


def uses_pyarrow_backend(constructor: Constructor | ConstructorEager) -> bool:
    """Checks if the pandas-like constructor uses pyarrow backend."""
    return constructor.__name__ in {
        "pandas_pyarrow_constructor",
        "modin_pyarrow_constructor",
    }
