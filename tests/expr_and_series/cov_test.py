from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 3], "b": [1, 2, 3], "c": [1, None, 1]}


def test_cov_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(
        a_b_sample=nw.cov("a", "b"),
        a_b_population=nw.cov("a", "b", ddof=0),
        a_b_ddof_2=nw.cov("a", "b", ddof=2),
        a_c_sample=nw.cov("a", "c"),
        a_c_population=nw.cov("a", "c", ddof=0),
        a_c_ddof_2=nw.cov("a", "c", ddof=2),
    )
    expected = {
        "a_b_sample": [1.0],
        "a_b_population": [0.6666666666666666],
        "a_b_ddof_2": [2.0],
        "a_c_sample": [0.0],
        "a_c_population": [0.0],
        "a_c_ddof_2": [None],
    }
    assert_equal_data(result, expected)


def test_cov_expr_inputs(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.cov(nw.col("a").alias("left"), nw.col("b").alias("right")),
        expr=nw.cov(nw.col("a") * nw.col("b"), nw.col("b") - 0.5),
    )
    expected = {"left": [1.0], "expr": [4.0]}
    assert_equal_data(result, expected)


def test_cov_single_valid_pair_population(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, None], "b": [2, None]}))
    result = df.select(cov=nw.cov("a", "b", ddof=0))
    expected = {"cov": [0.0]}
    assert_equal_data(result, expected)


def test_cov_invalid_denominator(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 3], "b": [1, 2]}))
    result = df.select(
        cov_den_zero=nw.cov("a", "b", ddof=2), cov_den_neg=nw.cov("a", "b", ddof=3)
    )
    expected = {"cov_den_zero": [None], "cov_den_neg": [None]}
    assert_equal_data(result, expected)
    if result.implementation.is_pyarrow():
        assert result.schema == {
            "cov_den_zero": nw.Float64(),
            "cov_den_neg": nw.Float64(),
        }
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        return

    result = df.with_columns(
        cov_den_zero=nw.cov("a", "b", ddof=2), cov_den_neg=nw.cov("a", "b", ddof=3)
    ).select("cov_den_zero", "cov_den_neg")
    expected = {"cov_den_zero": [None, None], "cov_den_neg": [None, None]}
    assert_equal_data(result, expected)


def test_cov_over(constructor: Constructor) -> None:
    if not any(x in str(constructor) for x in ("duckdb", "pyspark", "sqlframe")):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()

    df = nw.from_native(
        constructor({
            "i": [0, 1, 2, 3, 4],
            "g": [1, 1, 1, 2, 2],
            "a": [1, 3, 3, 2, 4],
            "b": [1, 2, 3, 1, 5],
        })
    )
    result = (
        df
        .with_columns(
            sample=nw.cov("a", "b").over("g"),
            population=nw.cov("a", "b", ddof=0).over("g"),
        )
        .sort("i")
        .select("sample", "population")
    )
    expected = {
        "sample": [1.0, 1.0, 1.0, 4.0, 4.0],
        "population": [2 / 3, 2 / 3, 2 / 3, 2.0, 2.0],
    }
    assert_equal_data(result, expected)


def test_cov_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        a_b_sample=nw.cov(df["a"], df["b"]),
        a_b_population=nw.cov(df["a"], df["b"], ddof=0),
        a_b_ddof_2=nw.cov(df["a"], df["b"], ddof=2),
        a_c_sample=nw.cov(df["a"], df["c"]),
    )
    expected = {
        "a_b_sample": [1.0],
        "a_b_population": [0.6666666666666666],
        "a_b_ddof_2": [2.0],
        "a_c_sample": [0.0],
    }
    assert_equal_data(result, expected)
