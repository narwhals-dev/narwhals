from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            [nw.col("a").min().alias("min"), nw.col("a", "b").mean()],
            {"min": [1], "a": [2], "b": [5]},
        ),
        ([(nw.col("a") + nw.col("b").max()).alias("x")], {"x": [7, 8, 9]}),
        ([nw.col("a"), nw.col("b").min()], {"a": [1, 2, 3], "b": [4, 4, 4]}),
        ([nw.col("a").max(), nw.col("b")], {"a": [3, 3, 3], "b": [4, 5, 6]}),
        (
            [nw.col("a"), nw.col("b").min().alias("min")],
            {"a": [1, 2, 3], "min": [4, 4, 4]},
        ),
    ],
    ids=range(5),
)
def test_scalar_reduction_select(
    constructor: Constructor,
    expr: list[Any],
    expected: dict[str, list[Any]],
    request: pytest.FixtureRequest,
) -> None:
    if "pyspark" in str(constructor) and request.node.callspec.id in {
        "pyspark-2",
        "pyspark-3",
        "pyspark-4",
    }:
        request.applymarker(pytest.mark.xfail)

    if "duckdb" in str(constructor) and request.node.callspec.id not in {"duckdb-0"}:
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))
    result = df.select(*expr)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            [nw.col("a").min().alias("min"), nw.col("a", "b").mean()],
            {"min": [1, 1, 1], "a": [2, 2, 2], "b": [5, 5, 5]},
        ),
        ([(nw.col("a") + nw.col("b").max()).alias("x")], {"x": [7, 8, 9]}),
        ([nw.col("a"), nw.col("b").min()], {"a": [1, 2, 3], "b": [4, 4, 4]}),
        ([nw.col("a").max(), nw.col("b")], {"a": [3, 3, 3], "b": [4, 5, 6]}),
        (
            [nw.col("a"), nw.col("b").min().alias("min")],
            {"a": [1, 2, 3], "min": [4, 4, 4]},
        ),
    ],
    ids=range(5),
)
def test_scalar_reduction_with_columns(
    constructor: Constructor,
    expr: list[Any],
    expected: dict[str, list[Any]],
    request: pytest.FixtureRequest,
) -> None:
    if "duckdb" in str(constructor) or (
        "pyspark" in str(constructor) and request.node.callspec.id != "pyspark-1"
    ):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(*expr).select(*expected.keys())
    assert_equal_data(result, expected)


def test_empty_scalar_reduction_select(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "pyspark" in str(constructor) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {
        "str": [*"abcde"],
        "int": [0, 1, 2, 3, 4],
        "bool": [True, False, False, True, False],
    }
    expressions = {
        "all": nw.col("bool").all(),
        "any": nw.col("bool").any(),
        "max": nw.col("int").max(),
        "mean": nw.col("int").mean(),
        "min": nw.col("int").min(),
        "sum": nw.col("int").sum(),
    }

    df = nw.from_native(constructor(data)).filter(str="z")

    result = df.select(**expressions)
    expected = {
        "all": [True],
        "any": [False],
        "max": [None],
        "mean": [None],
        "min": [None],
        "sum": [0],
    }
    assert_equal_data(result, expected)


def test_empty_scalar_reduction_with_columns(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "pyspark" in str(constructor) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    from itertools import chain

    data = {
        "str": [*"abcde"],
        "int": [0, 1, 2, 3, 4],
        "bool": [True, False, False, True, False],
    }
    expressions = {
        "all": nw.col("bool").all(),
        "any": nw.col("bool").any(),
        "max": nw.col("int").max(),
        "mean": nw.col("int").mean(),
        "min": nw.col("int").min(),
        "sum": nw.col("int").sum(),
    }

    df = nw.from_native(constructor(data)).filter(str="z")
    result = df.with_columns(**expressions)
    expected: dict[str, list[Any]] = {
        k: [] for k in chain(df.collect_schema(), expressions)
    }
    assert_equal_data(result, expected)


def test_empty_scalar_reduction_series(constructor_eager: ConstructorEager) -> None:
    data = {
        "str": [*"abcde"],
        "int": [0, 1, 2, 3, 4],
        "bool": [True, False, False, True, False],
    }
    df = nw.from_native(constructor_eager(data), eager_only=True).filter(str="z")
    result_s = {
        "all": [df["bool"].all()],
        "any": [df["bool"].any()],
        "max": [df["int"].max()],
        "mean": [df["int"].mean()],
        "min": [df["int"].min()],
        "sum": [df["int"].sum()],
    }
    expected = {
        "all": [True],
        "any": [False],
        "max": [None],
        "mean": [None],
        "min": [None],
        "sum": [0],
    }

    assert_equal_data(result_s, expected)
