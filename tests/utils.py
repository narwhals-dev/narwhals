from __future__ import annotations

import math
import os
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import narwhals as nw
from narwhals._utils import Implementation, parse_version, zip_strict
from narwhals.dependencies import get_pandas
from narwhals.translate import from_native

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd
    import pytest
    from pyspark.sql import SparkSession
    from sqlframe.duckdb import DuckDBSession
    from typing_extensions import TypeAlias

    from narwhals._native import NativeLazyFrame
    from narwhals.typing import Frame, IntoDataFrame, TimeUnit


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

Constructor: TypeAlias = Callable[[Any], "NativeLazyFrame | IntoDataFrame"]
ConstructorEager: TypeAlias = Callable[[Any], "IntoDataFrame"]
ConstructorLazy: TypeAlias = Callable[[Any], "NativeLazyFrame"]
ConstructorPandasLike: TypeAlias = Callable[[Any], "pd.DataFrame"]

NestedOrEnumDType: TypeAlias = "nw.List | nw.Array | nw.Struct | nw.Enum"
"""`DType`s which **cannot** be used as bare types."""

ID_PANDAS_LIKE = frozenset(
    ("pandas", "pandas[nullable]", "pandas[pyarrow]", "modin", "modin[pyarrow]", "cudf")
)
ID_CUDF = frozenset(("cudf",))
_CONSTRUCTOR_FIXTURE_NAMES = frozenset[str](
    ("constructor_eager", "constructor", "constructor_pandas_like")
)


def _to_comparable_list(column_values: Any) -> Any:
    if isinstance(column_values, nw.Series) and column_values.implementation.is_pyarrow():
        import pyarrow as pa

        if isinstance(column_values.to_native(), pa.Array):  # pragma: no cover
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


def is_pd_na(value: Any) -> bool:
    return (pd := get_pandas()) is not None and pd.isna(value)


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
        result = from_native(result.collect("pyarrow"))
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
                are_equivalent_values = lhs is None or is_pd_na(lhs) or math.isnan(lhs)
            elif lhs is None:
                are_equivalent_values = rhs is None
            elif isinstance(lhs, list) and isinstance(rhs, list):
                are_equivalent_values = all(
                    left_side == right_side for left_side, right_side in zip(lhs, rhs)
                )
            elif is_pd_na(lhs):
                are_equivalent_values = is_pd_na(rhs)
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
                f"Mismatch at index {i}, key {key}: {lhs} != {rhs}\nExpected: {expected}\nGot: {result}"
            )


def assert_equal_series(
    result: nw.Series[Any], expected: Sequence[Any], name: str
) -> None:
    assert_equal_data(result.to_frame(), {name: expected})


def sqlframe_session() -> DuckDBSession:
    from sqlframe.duckdb import DuckDBSession

    # NOTE: `__new__` override inferred by `pyright` only
    # https://github.com/eakmanrq/sqlframe/blob/772b3a6bfe5a1ffd569b7749d84bea2f3a314510/sqlframe/base/session.py#L181-L184
    return cast("DuckDBSession", DuckDBSession())  # type: ignore[redundant-cast]


def pyspark_session() -> SparkSession:  # pragma: no cover
    if is_spark_connect := os.environ.get("SPARK_CONNECT", None):
        from pyspark.sql.connect.session import SparkSession
    else:
        from pyspark.sql import SparkSession
    builder = cast("SparkSession.Builder", SparkSession.builder).appName("unit-tests")
    builder = (
        builder.remote(f"sc://localhost:{os.environ.get('SPARK_PORT', '15002')}")
        if is_spark_connect
        else builder.master("local[1]").config("spark.ui.enabled", "false")
    )
    return (
        builder.config("spark.default.parallelism", "1")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
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


def maybe_collect(df: Frame) -> Frame:
    """Collect to DataFrame if it is a LazyFrame.

    Use this function to test specific behaviors during collection.
    For example, Polars only errors when we call `collect` in the lazy case.
    """
    return df.collect() if isinstance(df, nw.LazyFrame) else df


def time_unit_compat(time_unit: TimeUnit, request: pytest.FixtureRequest, /) -> TimeUnit:
    """Replace `time_unit` with one that is supported by the requested backend."""
    if _CONSTRUCTOR_FIXTURE_NAMES.isdisjoint(request.fixturenames):  # pragma: no cover
        msg = (
            f"`time_unit_compat` requires the test function to use a `constructor*` fixture.\n"
            f"Hint:\n\n"
            f"Try adding one of these as a parameter:\n    {sorted(_CONSTRUCTOR_FIXTURE_NAMES)!r}"
        )
        raise NotImplementedError(msg)
    request_id = request.node.callspec.id
    if "duckdb" in request_id:
        return "us"
    pandas_like = ID_PANDAS_LIKE - ID_CUDF
    if PANDAS_VERSION < (2,) and any(name in request_id for name in pandas_like):
        return "ns"
    return time_unit
