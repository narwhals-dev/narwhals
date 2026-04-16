from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.testing.typing import Constructor, ConstructorEager

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
