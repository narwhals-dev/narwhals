from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from narwhals.exceptions import ShapeError
from tests.utils import (
    DUCKDB_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {"a": [1, 1, 2, 2, 3], "b": [1, 2, 3, 3, 4]}
data_nulls = {
    "null_mode": [None, None, 2.0],
    "value_mode": [2.0, 2.0, None],
    "tied": [None, 2.0, 3.0],
    "bool_null_mode": [None, None, True],
    "bool_value_mode": [True, True, None],
    "bool_no_nulls": [True, True, False],
}
data_group_nulls = {
    "grp": ["g1", "g1", "g1", "g2", "g2", "g2"],
    "vals": [None, None, 2.0, 3.0, None, 3.0],
}
data_group = {
    "grp": ["g1", "g1", "g1", "g1", "g2", "g2", "g2"],
    "vals_unimodal": [2, 1, 1, 3, 3, 3, 4],
    "vals_multimodal_num": [1, 1, 2, 2, 3, 3, 4],
    "vals_multimodal_str": ["foo", "foo", "bar", "bar", "baz", "baz", "foo"],
}


def test_mode_single_expr_keep_all(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").mode(keep="all")).sort("a")
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)


def test_mode_series_keep_all(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.mode(keep="all").sort()
    expected = {"a": [1, 2]}
    assert_equal_data({"a": result}, expected)


def test_mode_series_keep_any(constructor_eager: ConstructorEager) -> None:
    frame = nw.from_native(constructor_eager(data), eager_only=True)
    result_a = frame["a"].mode(keep="any")
    assert result_a in {1, 2}

    result_b = frame["b"].mode(keep="any")
    assert result_b == 3


def test_mode_different_lengths_keep_all(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < PkgVersion("1.10"):
        pytest.skip()
    df = nw.from_native(constructor_eager(data))
    with pytest.raises(ShapeError):
        df.select(nw.col("a", "b").mode(keep="all"))


def test_mode_expr_keep_any(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b").mode(keep="any")).lazy().collect()

    try:
        expected = {"a": [1], "b": [3]}
        assert_equal_data(result, expected)
    except AssertionError:  # pragma: no cover
        expected = {"a": [2], "b": [3]}
        assert_equal_data(result, expected)


def test_mode_expr_keep_all_lazy(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    impl = df.implementation
    not_implemented = {
        nw.Implementation.DUCKDB,
        nw.Implementation.IBIS,
        nw.Implementation.PYSPARK,
        nw.Implementation.PYSPARK_CONNECT,
        nw.Implementation.SQLFRAME,
    }
    msg = re.escape("`Expr.mode(keep='all')` is not implemented for backend")
    context = (
        pytest.raises(NotImplementedError, match=msg)
        if impl in not_implemented
        else does_not_raise()
    )

    with context:
        result = df.select(nw.col("a").mode(keep="all").sum())
        assert_equal_data(result, {"a": [3]})


def test_mode_group_by_unimodal(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    df = nw.from_native(constructor(data_group))
    impl = df.implementation

    if impl.is_dask() or impl.is_pyarrow():
        # Issue tracker:
        #   - Dask: https://github.com/narwhals-dev/narwhals/pull/3019#issuecomment-3216649862
        #   - PyArrow: https://github.com/apache/arrow/issues/20359
        request.applymarker(pytest.mark.xfail)

    result = (
        df.group_by("grp")
        .agg(nw.col("vals_unimodal").mode(keep="any"))
        .sort("grp")
        .lazy()
        .collect()
    )
    expected = {"grp": ["g1", "g2"], "vals_unimodal": [1, 3]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("mode_col", "expected_opt_1", "expected_opt_2"),
    [
        ("vals_multimodal_num", [1, 3], [2, 3]),
        ("vals_multimodal_str", ["foo", "baz"], ["bar", "baz"]),
    ],
)
def test_mode_group_by_multimodal(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    mode_col: str,
    expected_opt_1: list[Any],
    expected_opt_2: list[Any],
) -> None:
    df = nw.from_native(constructor(data_group))
    impl = df.implementation

    if impl.is_dask() or impl.is_pyarrow():
        # Issue tracker:
        #   - Dask: https://github.com/narwhals-dev/narwhals/pull/3019#issuecomment-3216649862
        #   - PyArrow: https://github.com/apache/arrow/issues/20359
        request.applymarker(pytest.mark.xfail)

    result = (
        df.group_by("grp")
        .agg(nw.col(mode_col).mode(keep="any"))
        .sort("grp")
        .lazy()
        .collect()
    )
    try:
        expected = {"grp": ["g1", "g2"], mode_col: expected_opt_1}
        assert_equal_data(result, expected)
    except AssertionError:  # pragma: no cover
        expected = {"grp": ["g1", "g2"], mode_col: expected_opt_2}
        assert_equal_data(result, expected)


@pytest.mark.parametrize(
    "mode_expr",
    [
        nw.col("vals_unimodal", "vals_multimodal_num").mode(keep="any"),
        [
            nw.col("vals_unimodal").mode(keep="any"),
            nw.col("vals_multimodal_num").mode(keep="any"),
        ],
    ],
)
def test_mode_group_by_multiple_cols(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    mode_expr: nw.Expr | list[nw.Expr],
) -> None:
    df = nw.from_native(constructor(data_group))
    impl = df.implementation

    if impl.is_dask() or impl.is_pyarrow():
        # Issue tracker:
        #   - Dask: https://github.com/narwhals-dev/narwhals/pull/3019#issuecomment-3216649862
        #   - PyArrow: https://github.com/apache/arrow/issues/20359
        request.applymarker(pytest.mark.xfail)

    result = df.group_by("grp").agg(mode_expr).sort("grp").lazy().collect()

    try:
        expected = {
            "grp": ["g1", "g2"],
            "vals_unimodal": [1, 3],
            "vals_multimodal_num": [1, 3],
        }
        assert_equal_data(result, expected)
    except AssertionError:  # pragma: no cover
        expected = {
            "grp": ["g1", "g2"],
            "vals_unimodal": [1, 3],
            "vals_multimodal_num": [2, 3],
        }
        assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("column", "expected"),
    [("null_mode", [None]), ("value_mode", [2.0]), ("tied", [2.0, 3.0, None])],
)
def test_mode_series_nulls(
    constructor_eager: ConstructorEager, column: str, expected: list[float | None]
) -> None:
    series = nw.from_native(constructor_eager(data_nulls), eager_only=True)[column]
    result = series.mode(keep="all").sort(nulls_last=True)
    assert_equal_data({column: result}, {column: expected})


@pytest.mark.parametrize(
    ("column", "expected"),
    [("bool_null_mode", None), ("bool_value_mode", True), ("bool_no_nulls", True)],
)
def test_mode_series_nulls_boolean(
    constructor_eager: ConstructorEager, column: str, *, expected: bool | None
) -> None:
    # pandas' nullable booleans take a separate path (pandas-dev/pandas#70487).
    series = nw.from_native(constructor_eager(data_nulls), eager_only=True)[column]
    assert_equal_data({column: series.mode(keep="all")}, {column: [expected]})


@pytest.mark.parametrize(
    ("column", "expected"), [("null_mode", None), ("value_mode", 2.0)]
)
def test_mode_expr_keep_any_nulls(
    constructor: Constructor, column: str, expected: float | None
) -> None:
    df = nw.from_native(constructor(data_nulls))
    result = df.select(nw.col(column).mode(keep="any")).lazy().collect()
    assert_equal_data(result, {column: [expected]})


def test_mode_group_by_nulls(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    df = nw.from_native(constructor(data_group_nulls))
    impl = df.implementation

    if impl.is_dask() or impl.is_pyarrow():
        # Issue tracker:
        #   - Dask: https://github.com/narwhals-dev/narwhals/pull/3019#issuecomment-3216649862
        #   - PyArrow: https://github.com/apache/arrow/issues/20359
        request.applymarker(pytest.mark.xfail)

    result = (
        df.group_by("grp")
        .agg(nw.col("vals").mode(keep="any"))
        .sort("grp")
        .lazy()
        .collect()
    )
    assert_equal_data(result, {"grp": ["g1", "g2"], "vals": [None, 3.0]})


def test_mode_over_nulls(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < PkgVersion("1.3"):
        pytest.skip(reason="`over` requires DuckDB 1.3.0")
    df = nw.from_native(constructor(data_group_nulls))
    impl = df.implementation
    if impl.is_pandas_like() or impl.is_pyarrow() or impl.is_dask():
        reason = "`mode` is not supported in `over` for pandas-like, PyArrow and Dask."
        request.applymarker(pytest.mark.xfail(reason=reason))
    result = df.with_columns(nw.col("vals").mode(keep="any").over("grp")).sort(
        "grp", "vals", nulls_last=True
    )
    expected = {
        "grp": ["g1", "g1", "g1", "g2", "g2", "g2"],
        "vals": [None, None, None, 3.0, 3.0, 3.0],
    }
    assert_equal_data(result, expected)
