from __future__ import annotations

import re

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data, maybe_collect

# Backends differ in their algorithm, so compare with the exact value within a tolerance.
TOLERANCE = 2.0
N = 1000
data = {"a": list(range(N)), "z": [float(i) for i in range(N)]}


@pytest.mark.parametrize(("q", "expected"), [(0.1, 99.9), (0.5, 499.5), (0.9, 899.1)])
def test_approx_quantile_expr(
    constructor: Constructor, q: float, expected: float, request: pytest.FixtureRequest
) -> None:
    if "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="single partition only"))
    df = nw.from_native(constructor(data)).lazy()
    result = df.select(nw.col("a", "z").approx_quantile(q)).collect()
    assert result["a"].dtype.is_float()
    assert result["a"].item() == pytest.approx(expected, abs=TOLERANCE)
    assert result["z"].item() == pytest.approx(expected, abs=TOLERANCE)


def test_approx_quantile_int_not_truncated(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="single partition only"))
    df = nw.from_native(constructor({"a": [1, 2]})).lazy()
    result = df.select(nw.col("a").approx_quantile(1.0)).collect()
    assert_equal_data(result, {"a": [2.0]})


def test_approx_quantile_ignores_nulls(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="single partition only"))
    df = nw.from_native(constructor({"a": [None, 1.0, 1.0, None]})).lazy()
    result = df.select(nw.col("a").approx_quantile(0.5)).collect()
    assert_equal_data(result, {"a": [1.0]})


def test_approx_quantile_all_null(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="single partition only"))
    df = nw.from_native(constructor({"a": [1.0, None]})).lazy()
    result = df.filter(nw.col("a").is_null()).select(nw.col("a").approx_quantile(0.5))
    assert_equal_data(result.collect(), {"a": [None]})


def test_approx_quantile_out_of_bounds_raises(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1.0, 2.0, 3.0]}))
    expr = nw.col("a").approx_quantile(1.5)
    # Broad `Exception`: error types differ per backend.
    with pytest.raises(Exception):  # noqa: B017, PT011
        maybe_collect(df.select(expr))


def test_approx_quantile_group_by(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="no group-by quantile in Dask"))
    df = nw.from_native(constructor({"g": [1, 1, 2], "a": [2.0, 2.0, 4.0]})).lazy()
    result = df.group_by("g").agg(nw.col("a").approx_quantile(0.5)).sort("g").collect()
    assert_equal_data(result, {"g": [1, 2], "a": [2.0, 4.0]})


def test_approx_quantile_group_by_two_quantiles_same_column(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="no group-by quantile in Dask"))
    df = nw.from_native(constructor({"g": [1, 1, 1], "a": [1.0, 2.0, 3.0]})).lazy()
    result = (
        df.group_by("g")
        .agg(
            nw.col("a").approx_quantile(0.0).alias("low"),
            nw.col("a").approx_quantile(1.0).alias("high"),
        )
        .collect()
    )
    assert_equal_data(result, {"g": [1], "low": [1.0], "high": [3.0]})


def test_approx_quantile_group_by_all_null_group(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="no group-by quantile in Dask"))
    df = nw.from_native(constructor({"g": [1, 2], "a": [None, 1.0]})).lazy()
    result = df.group_by("g").agg(nw.col("a").approx_quantile(0.5)).sort("g").collect()
    assert_equal_data(result, {"g": [1, 2], "a": [None, 1.0]})


def test_approx_quantile_over(constructor: Constructor) -> None:
    df = nw.from_native(
        constructor({"i": [0, 1, 2], "g": [1, 1, 2], "a": [2.0, 2.0, 4.0]})
    ).lazy()
    result = (
        df.with_columns(nw.col("a").approx_quantile(0.5).over("g").alias("q"))
        .select("i", "q")
        .sort("i")
        .collect()
    )
    assert_equal_data(result, {"i": [0, 1, 2], "q": [2.0, 2.0, 4.0]})


def test_approx_quantile_dask_multiple_partitions_raises(
    constructor: Constructor,
) -> None:
    if "dask_lazy_p2" not in str(constructor):
        pytest.skip()
    df = nw.from_native(constructor(data)).lazy()
    msg = re.escape("not supported for Dask backend with multiple partitions")
    with pytest.raises(NotImplementedError, match=msg):
        df.select(nw.col("a").approx_quantile(0.5)).collect()


@pytest.mark.parametrize(("q", "expected"), [(0.1, 99.9), (0.5, 499.5), (0.9, 899.1)])
def test_approx_quantile_series(
    constructor_eager: ConstructorEager, q: float, expected: float
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    assert series.approx_quantile(q) == pytest.approx(expected, abs=TOLERANCE)
