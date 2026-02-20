from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [1, 2, 3]}


@pytest.mark.parametrize(
    ("output_name", "a", "b", "expected_corr"),
    [
        ("a", "a", "b", 0.5),
        ("b", nw.col("a").alias("b"), nw.col("b").alias("c"), 0.5),
        ("a", nw.col("a") * nw.col("b"), nw.col("b") - 0.5, 0.8660254037844386),
    ],
)
def test_corr_expr(
    constructor: Constructor,
    output_name: str,
    a: str | nw.Expr,
    b: str | nw.Expr,
    expected_corr: float,
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.corr(a, b))
    expected = {output_name: [expected_corr]}
    assert_equal_data(result, expected)
    result = df.select(result=nw.corr(a, b))
    expected = {"result": [expected_corr]}
    assert_equal_data(result, expected)


def test_corr_expr_spearman(constructor: Constructor) -> None:
    data = {"a": [1, 6, 2, 3, 3], "b": [1, 1, 1, 3, 3]}
    context = (
        does_not_raise()
        if any(x in str(constructor) for x in ("pandas", "polars", "modin", "cudf"))
        else pytest.raises(NotImplementedError)
    )
    df = nw.from_native(constructor(data))
    with context:
        result = df.select(result=nw.corr("a", "b", method="spearman"))
        expected = {"result": [0.29617443887954614]}
        assert_equal_data(result, expected)


def test_corr_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(corr=nw.corr(df["a"], df["b"]))
    expected = {"corr": [0.5]}
    assert_equal_data(result, expected)
