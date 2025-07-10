# Test assorted functions which we overwrite in stable.v2

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

import narwhals.stable.v2 as nw_v2
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

if TYPE_CHECKING:
    from typing_extensions import assert_type

    from tests.utils import Constructor, ConstructorEager


def test_toplevel(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1,):
        pytest.skip()
    df = nw_v2.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, None, 9]})
    )
    result = df.select(
        min=nw_v2.min("a"),
        max=nw_v2.max("a"),
        mean=nw_v2.mean("a"),
        median=nw_v2.median("a"),
        sum=nw_v2.sum("a"),
        sum_h=nw_v2.sum_horizontal("a"),
        min_h=nw_v2.min_horizontal("a"),
        max_h=nw_v2.max_horizontal("a"),
        mean_h=nw_v2.mean_horizontal("a"),
        len=nw_v2.len(),
        concat_str=nw_v2.concat_str(nw_v2.lit("a"), nw_v2.lit("b")),
        any_h=nw_v2.any_horizontal(nw_v2.lit(True), nw_v2.lit(True), ignore_nulls=True),  # noqa: FBT003
        all_h=nw_v2.all_horizontal(nw_v2.lit(True), nw_v2.lit(True), ignore_nulls=True),  # noqa: FBT003
        first=nw_v2.nth(0),
        no_first=nw_v2.exclude("a", "c"),
        coalesce=nw_v2.coalesce("c", "a"),
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
        "coalesce": [7, 2, 9],
    }
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)


def test_when_then(constructor_eager: ConstructorEager) -> None:
    df = nw_v2.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [4, 5, 6], "c": [6, 7, 8]})
    )
    result = df.select(nw_v2.when(nw_v2.col("a") > 1).then("b").otherwise("c"))
    expected = {"b": [6, 5, 6]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)


def test_constructors() -> None:
    pytest.importorskip("pyarrow")
    if PANDAS_VERSION < (2, 2):
        pytest.skip()
    assert nw_v2.new_series("a", [1, 2, 3], backend="pandas").to_list() == [1, 2, 3]
    arr: np.ndarray[tuple[int, int], Any] = np.array([[1, 2], [3, 4]])  # pyright: ignore[reportAssignmentType]
    result = nw_v2.from_numpy(arr, schema=["a", "b"], backend="pandas")
    assert_equal_data(result, {"a": [1, 3], "b": [2, 4]})
    assert isinstance(result, nw_v2.DataFrame)
    result = nw_v2.from_numpy(
        arr,
        schema=nw_v2.Schema({"a": nw_v2.Int64(), "b": nw_v2.Int64()}),
        backend="pandas",
    )
    assert_equal_data(result, {"a": [1, 3], "b": [2, 4]})
    assert isinstance(result, nw_v2.DataFrame)
    result = nw_v2.from_dict({"a": [1, 2, 3]}, backend="pandas")
    assert_equal_data(result, {"a": [1, 2, 3]})
    assert isinstance(result, nw_v2.DataFrame)
    result = nw_v2.from_arrow(pd.DataFrame({"a": [1, 2, 3]}), backend="pandas")
    assert_equal_data(result, {"a": [1, 2, 3]})
    assert isinstance(result, nw_v2.DataFrame)


def test_join(constructor_eager: ConstructorEager) -> None:
    df = nw_v2.from_native(constructor_eager({"a": [1, 2, 3]})).lazy()
    result = df.join(df, how="inner", on="a").sort("a")  # type: ignore[arg-type]
    expected = {"a": [1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.LazyFrame)
    result_eager = df.collect().join(df.collect(), how="inner", on="a")
    assert_equal_data(result_eager, expected)
    assert isinstance(result_eager, nw_v2.DataFrame)


def test_by_name(constructor_eager: ConstructorEager) -> None:
    df = nw_v2.from_native(constructor_eager({"a": [1, 2, 3]})).lazy()
    result = df.select(nw_v2.col("a").alias("b"), "a")
    expected = {"b": [1, 2, 3], "a": [1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.LazyFrame)
    result_eager = df.collect().select(nw_v2.col("a").alias("b"), "a")
    assert_equal_data(result_eager, expected)
    assert isinstance(result_eager, nw_v2.DataFrame)


def test_values_counts_v2(constructor_eager: ConstructorEager) -> None:
    df = nw_v2.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df["a"].value_counts().sort("a")
    expected = {"a": [1, 2, 3], "count": [1, 1, 1]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)


def test_is_duplicated_unique(constructor_eager: ConstructorEager) -> None:
    df = nw_v2.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    assert df.is_duplicated().to_list() == [False, False, False]
    assert df.is_unique().to_list() == [True, True, True]
    assert isinstance(df.is_duplicated(), nw_v2.Series)
    assert isinstance(df.is_unique(), nw_v2.Series)


def test_concat(constructor_eager: ConstructorEager) -> None:
    df = nw_v2.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = nw_v2.concat([df, df], how="vertical")
    expected = {"a": [1, 2, 3, 1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)
    if TYPE_CHECKING:
        assert_type(result, nw_v2.DataFrame[Any])


@pytest.mark.filterwarnings(
    "ignore:.*all arguments of to_dict except for the argument:FutureWarning"
)
def test_to_dict(constructor_eager: ConstructorEager) -> None:
    df = nw_v2.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.to_dict(as_series=False)
    expected = {"a": [1, 2, 3]}
    assert result == expected


def test_tail(constructor_eager: ConstructorEager) -> None:
    df = nw_v2.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True).lazy()
    result = df.tail(3)
    expected = {"a": [1, 2, 3]}
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(
    "ignore:.*all arguments of to_dict except for the argument:FutureWarning"
)
def test_to_dict_as_series(constructor_eager: ConstructorEager) -> None:
    df = nw_v2.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.to_dict(as_series=True)
    expected = {"a": [1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result["a"], nw_v2.Series)


@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_v2(constructor_eager: ConstructorEager) -> None:
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (13,):
        pytest.skip()
    if "cudf" in str(constructor_eager):
        pytest.skip()
    df = nw_v2.from_native(constructor_eager({"a": [1, 1, 2]}), eager_only=True)
    result = df["a"].hist(bins=[-1, 1, 2])
    expected = {"breakpoint": [1, 2], "count": [2, 1]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)


def test_int_select_pandas() -> None:
    df = nw_v2.from_native(pd.DataFrame({0: [1, 2], "b": [3, 4]}))
    with pytest.raises(
        nw_v2.exceptions.InvalidIntoExprError, match="\n\nHint:\n- if you were trying"
    ):
        nw_v2.to_native(df.select(0))  # type: ignore[arg-type]
    with pytest.raises(
        nw_v2.exceptions.InvalidIntoExprError, match="\n\nHint:\n- if you were trying"
    ):
        nw_v2.to_native(df.lazy().select(0))  # type: ignore[arg-type]


def test_any_horizontal() -> None:
    # here, it defaults to Kleene logic.
    pytest.importorskip("polars")
    import polars as pl

    df = nw_v2.from_native(
        pl.DataFrame({"a": [True, True, False], "b": [True, None, None]})
    )
    result = df.select(nw_v2.any_horizontal("a", "b", ignore_nulls=True))
    expected = {"a": [True, True, False]}
    assert_equal_data(result, expected)


def test_all_horizontal() -> None:
    # here, it defaults to Kleene logic.
    pytest.importorskip("polars")
    import polars as pl

    df = nw_v2.from_native(
        pl.DataFrame({"a": [True, True, False], "b": [True, None, None]})
    )
    result = df.select(nw_v2.all_horizontal("a", "b", ignore_nulls=True))
    expected = {"a": [True, True, False]}
    assert_equal_data(result, expected)


def test_with_row_index(constructor: Constructor) -> None:
    data = {"abc": ["foo", "bars"], "xyz": [100, 200], "const": [42, 42]}

    frame = nw_v2.from_native(constructor(data))
    result = frame.with_row_index(order_by="xyz")
    expected = {"index": [0, 1], **data}
    assert_equal_data(result, expected)
