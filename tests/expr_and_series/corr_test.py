from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 3], "b": [1, 2, 3], "c": [1, None, 1]}


@pytest.mark.parametrize(
    ("output_name", "a", "b", "expected_corr"),
    [
        ("a", "a", "b", 0.87),
        ("a", "a", "c", None),
        ("b", nw.col("a").alias("b"), nw.col("b").alias("c"), 0.87),
        ("a", nw.col("a") * nw.col("b"), nw.col("b") - 0.5, 0.99),
    ],
)
def test_corr_expr(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    output_name: str,
    a: str | nw.Expr,
    b: str | nw.Expr,
    expected_corr: float | None,
) -> None:
    if "pyspark" in str(constructor) and expected_corr is None:
        request.applymarker(
            pytest.skip(reason="Pyspark corr function does not allow None values")
        )
    df = nw.from_native(constructor(data))
    result = df.select(nw.corr(a, b).round(2))
    expected = {output_name: [expected_corr]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("output_name", "a", "b", "expected_corr"),
    [
        ("a", "a", "b", 0.87),
        ("a", "a", "c", None),
        ("b", nw.col("a").alias("b"), nw.col("b").alias("c"), 0.87),
        ("a", nw.col("a") * nw.col("b"), nw.col("b") - 0.5, 1),
    ],
)
def test_corr_expr_spearman(
    constructor: Constructor,
    output_name: str,
    a: str | nw.Expr,
    b: str | nw.Expr,
    expected_corr: float | None,
) -> None:
    context = (
        does_not_raise()
        if any(x in str(constructor) for x in ("pandas", "polars", "modin", "cudf"))
        else pytest.raises(NotImplementedError)
    )
    df = nw.from_native(constructor(data))
    with context:
        result = df.select(nw.corr(a, b, method="spearman").round(2))
        expected = {output_name: [expected_corr]}
        assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("output_name", "a", "b", "expected_corr"),
    [("a", "a", "b", 0.87), ("a", "a", "c", None)],
)
def test_corr_series(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    output_name: str,
    a: str,
    b: str,
    expected_corr: float | None,
) -> None:
    if "pyspark" in str(constructor_eager) and expected_corr is None:
        request.applymarker(
            pytest.skip(reason="Pyspark corr function does not allow None values")
        )
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.corr(df[a], df[b]).round(2))
    expected = {output_name: [expected_corr]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("output_name", "a", "b", "expected_corr"),
    [("a", "a", "b", 0.87), ("a", "a", "c", None)],
)
def test_corr_series_spearman(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    output_name: str,
    a: str,
    b: str,
    expected_corr: float | None,
) -> None:
    if "pyspark" in str(constructor_eager) and expected_corr is None:
        request.applymarker(
            pytest.skip(reason="Pyspark corr function does not allow None values")
        )
    context = (
        does_not_raise()
        if any(
            x in str(constructor_eager)
            for x in ("pandas", "polars", "modin", "cudf", "pyarrow")
        )
        else pytest.raises(NotImplementedError)
    )
    df = nw.from_native(constructor_eager(data))
    with context:
        result = df.select(nw.corr(df[a], df[b]).round(2))
        expected = {output_name: [expected_corr]}
        assert_equal_data(result, expected)


def test_corr_over(constructor: Constructor) -> None:
    # Regression test for the window-broadcast path: `corr` must compose with
    # `over`/`with_columns` on SQL backends (mirrors `test_cov_over`).
    if not any(x in str(constructor) for x in ("duckdb", "pyspark", "sqlframe")):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()

    df = nw.from_native(
        constructor(
            {
                "i": [0, 1, 2, 3, 4],
                "g": [1, 1, 1, 2, 2],
                "a": [1, 3, 3, 2, 4],
                "b": [1, 2, 3, 1, 5],
            }
        )
    )
    result = df.with_columns(corr=nw.corr("a", "b").over("g")).sort("i").select("corr")
    # g=1: corr([1,3,3], [1,2,3]) = sqrt(3)/2; g=2: two points are perfectly correlated.
    expected = {"corr": [3**0.5 / 2, 3**0.5 / 2, 3**0.5 / 2, 1.0, 1.0]}
    assert_equal_data(result, expected)


def test_corr_pairwise_nulls(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    # Correlation is pairwise: a null in one column must drop that row entirely,
    # otherwise the other column's mean/stddev are computed over the wrong rows.
    # Regression test for the pyarrow path, which previously skipped this filtering.
    if "pyspark" in str(constructor):
        request.applymarker(
            pytest.skip(reason="PySpark corr function does not allow None values")
        )
    df = nw.from_native(
        constructor({"a": [1.0, 2.0, 3.0, 100.0], "b": [1.0, 2.0, 3.0, None]})
    )
    result = df.select(nw.corr("a", "b").alias("c"))
    # Pairwise-valid rows are a=[1, 2, 3], b=[1, 2, 3]: perfectly correlated.
    expected = {"c": [1.0]}
    assert_equal_data(result, expected)
