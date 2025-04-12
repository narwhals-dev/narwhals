# Test assorted functions which we overwrite in stable.v1
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import narwhals.stable.v1 as nw_v1
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_toplevel(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.select(
        min=nw_v1.min("a"),
        max=nw_v1.max("a"),
        mean=nw_v1.mean("a"),
        median=nw_v1.median("a"),
        sum=nw_v1.sum("a"),
        sum_h=nw_v1.sum_horizontal("a"),
        min_h=nw_v1.min_horizontal("a"),
        max_h=nw_v1.max_horizontal("a"),
        mean_h=nw_v1.mean_horizontal("a"),
        len=nw_v1.len(),
        concat_str=nw_v1.concat_str(nw_v1.lit("a"), nw_v1.lit("b")),
        any_h=nw_v1.any_horizontal(nw_v1.lit(True), nw_v1.lit(True)),  # noqa: FBT003
        all_h=nw_v1.all_horizontal(nw_v1.lit(True), nw_v1.lit(True)),  # noqa: FBT003
        first=nw_v1.nth(0),
        no_first=nw_v1.exclude("a"),
    )
    expected = {
        "min": [1, 1, 1],
        "max": [3, 3, 3],
        "mean": [2.0, 2.0, 2.0],
        "median": [2.0, 2.0, 2.0],
        "sum": [6, 6, 6],
        "sum_h": [1, 2, 3],
        "min_h": [1, 2, 3],
        "max_h": [1, 2, 3],
        "mean_h": [1, 2, 3],
        "len": [3, 3, 3],
        "concat_str": ["ab", "ab", "ab"],
        "any_h": [True, True, True],
        "all_h": [True, True, True],
        "first": [1, 2, 3],
        "no_first": [4, 5, 6],
    }
    assert_equal_data(result, expected)


def test_when_then(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [4, 5, 6], "c": [6, 7, 8]})
    )
    result = df.select(nw_v1.when(nw_v1.col("a") > 1).then("b").otherwise("c"))
    expected = {"b": [6, 5, 6]}
    assert_equal_data(result, expected)


def test_constructors() -> None:
    assert nw_v1.new_series("a", [1, 2, 3], backend="pandas").to_list() == [1, 2, 3]
    arr: np.ndarray[tuple[int, int], Any] = np.array([[1, 2], [3, 4]])  # pyright: ignore[reportAssignmentType]
    assert_equal_data(
        nw_v1.from_numpy(arr, schema=["a", "b"], backend="pandas"),
        {"a": [1, 3], "b": [2, 4]},
    )
    assert_equal_data(
        nw_v1.from_dict({"a": [1, 2, 3]}, backend="pandas"), {"a": [1, 2, 3]}
    )
    assert_equal_data(
        nw_v1.from_arrow(pd.DataFrame({"a": [1, 2, 3]}), backend="pandas"),
        {"a": [1, 2, 3]},
    )


def test_join(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]})).lazy()
    result = df.join(df, how="inner", on="a")  # type: ignore[arg-type]
    expected = {"a": [1, 2, 3]}
    assert_equal_data(result, expected)
    result_eager = df.collect().join(df.collect(), how="inner", on="a")
    assert_equal_data(result_eager, expected)


def test_by_name(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]})).lazy()
    result = df.select(nw_v1.col("a").alias("b"), "a")
    expected = {"b": [1, 2, 3], "a": [1, 2, 3]}
    assert_equal_data(result, expected)
    result_eager = df.collect().select(nw_v1.col("a").alias("b"), "a")
    assert_equal_data(result_eager, expected)


def test_values_counts_v1(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df["a"].value_counts().sort("a")
    expected = {"a": [1, 2, 3], "count": [1, 1, 1]}
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature.",
)
def test_hist_v1(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 1, 2]}), eager_only=True)
    result = df["a"].hist(bins=[-1, 1, 2])
    expected = {"breakpoint": [1, 2], "count": [2, 1]}
    assert_equal_data(result, expected)
