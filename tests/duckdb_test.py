"""PySpark support in Narwhals is still _very_ limited.

Start with a simple test file whilst we develop the basics.
Once we're a bit further along, we can integrate PySpark tests into the main test suite.
"""

from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING
from typing import Any

import pytest

import narwhals.stable.v1 as nw
from narwhals.exceptions import ColumnNotFoundError
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame

import polars as pl

duckdb = pytest.importorskip("duckdb")


def duckdb_constructor(obj: dict[str, Any]) -> IntoFrame:
    _df = pl.DataFrame(obj)
    return duckdb.table("_df")  # type: ignore[no-any-return]


# copied from tests/translate/from_native_test.py
def test_series_only() -> None:
    obj = duckdb_constructor({"a": [1, 2, 3]})
    with pytest.raises(TypeError, match="Cannot only use `series_only"):
        _ = nw.from_native(obj, series_only=True)


def test_eager_only_lazy() -> None:
    dframe = duckdb_constructor({"a": [1, 2, 3]})
    with pytest.raises(TypeError, match="Cannot only use .*`eager_only"):
        _ = nw.from_native(dframe, eager_only=True)


# copied from tests/frame/with_columns_test.py
def test_columns() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(duckdb_constructor(data))
    result = df.columns
    expected = ["a", "b", "z"]
    assert result == expected


# copied from tests/frame/with_columns_test.py
def test_with_columns_order() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(duckdb_constructor(data))
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.collect_schema().names() == ["a", "b", "z", "d"]
    expected = {"a": [2, 4, 3], "b": [4, 4, 6], "z": [7.0, 8, 9], "d": [0, 2, 1]}
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings("ignore:If `index_col` is not specified for `to_spark`")
def test_with_columns_empty() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(duckdb_constructor(data))
    result = df.select().with_columns()
    assert_equal_data(result, {})


def test_with_columns_order_single_row() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9], "i": [0, 1, 2]}
    df = nw.from_native(duckdb_constructor(data)).filter(nw.col("i") < 1).drop("i")
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.collect_schema().names() == ["a", "b", "z", "d"]
    expected = {"a": [2], "b": [4], "z": [7.0], "d": [0]}
    assert_equal_data(result, expected)


# copied from tests/frame/select_test.py
def test_select() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(duckdb_constructor(data))
    result = df.select("a")
    expected = {"a": [1, 3, 2]}
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings("ignore:If `index_col` is not specified for `to_spark`")
@pytest.mark.xfail
def test_empty_select() -> None:
    result = nw.from_native(duckdb_constructor({"a": [1, 2, 3]})).lazy().select()
    assert result.collect().shape == (0, 0)


# copied from tests/frame/filter_test.py
def test_filter() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(duckdb_constructor(data))
    result = df.filter(nw.col("a") > 1)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    assert_equal_data(result, expected)


# copied from tests/frame/schema_test.py
@pytest.mark.filterwarnings("ignore:Determining|Resolving.*")
def test_schema() -> None:
    df = nw.from_native(
        duckdb_constructor({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]})
    )
    result = df.schema
    expected = {"a": nw.Int64, "b": nw.Int64, "z": nw.Float64}

    result = df.schema
    assert result == expected
    result = df.lazy().collect().schema
    assert result == expected


def test_collect_schema() -> None:
    df = nw.from_native(
        duckdb_constructor({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]})
    )
    expected = {"a": nw.Int64, "b": nw.Int64, "z": nw.Float64}

    result = df.collect_schema()
    assert result == expected
    result = df.lazy().collect().collect_schema()
    assert result == expected


# copied from tests/frame/drop_test.py
@pytest.mark.parametrize(
    ("to_drop", "expected"),
    [
        ("abc", ["b", "z"]),
        (["abc"], ["b", "z"]),
        (["abc", "b"], ["z"]),
    ],
)
def test_drop(to_drop: list[str], expected: list[str]) -> None:
    data = {"abc": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(duckdb_constructor(data))
    assert df.drop(to_drop).collect_schema().names() == expected
    if not isinstance(to_drop, str):
        assert df.drop(*to_drop).collect_schema().names() == expected


@pytest.mark.parametrize(
    ("strict", "context"),
    [
        (True, pytest.raises(ColumnNotFoundError, match="z")),
        (False, does_not_raise()),
    ],
)
def test_drop_strict(context: Any, *, strict: bool) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6]}
    to_drop = ["a", "z"]

    df = nw.from_native(duckdb_constructor(data))

    with context:
        names_out = df.drop(to_drop, strict=strict).collect_schema().names()
        assert names_out == ["b"]


# copied from tests/frame/head_test.py
def test_head() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    expected = {"a": [1, 3], "b": [4, 4], "z": [7.0, 8.0]}

    df_raw = duckdb_constructor(data)
    df = nw.from_native(df_raw)

    result = df.head(2)
    assert_equal_data(result, expected)

    result = df.head(2)
    assert_equal_data(result, expected)

    # negative indices not allowed for lazyframes
    result = df.lazy().collect().head(-1)
    assert_equal_data(result, expected)


# copied from tests/frame/sort_test.py
def test_sort() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(duckdb_constructor(data))
    result = df.sort("a", "b")
    expected = {
        "a": [1, 2, 3],
        "b": [4, 6, 4],
        "z": [7.0, 9.0, 8.0],
    }
    assert_equal_data(result, expected)
    result = df.sort("a", "b", descending=[True, False]).lazy().collect()
    expected = {
        "a": [3, 2, 1],
        "b": [4, 6, 4],
        "z": [8.0, 9.0, 7.0],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("nulls_last", "expected"),
    [
        (True, {"a": [0, 2, 0, -1], "b": [3, 2, 1, float("nan")]}),
        (False, {"a": [-1, 0, 2, 0], "b": [float("nan"), 3, 2, 1]}),
    ],
)
def test_sort_nulls(*, nulls_last: bool, expected: dict[str, float]) -> None:
    data = {"a": [0, 0, 2, -1], "b": [1, 3, 2, None]}
    df = nw.from_native(duckdb_constructor(data))
    result = df.sort("b", descending=True, nulls_last=nulls_last).lazy().collect()
    assert_equal_data(result, expected)


# copied from tests/frame/add_test.py
@pytest.mark.xfail
def test_add() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(duckdb_constructor(data))
    result = df.with_columns(
        c=nw.col("a") + nw.col("b"),
        d=nw.col("a") - nw.col("a").mean(),
        e=nw.col("a") - nw.col("a").std(),
    )
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "c": [5, 7, 8],
        "d": [-1.0, 1.0, 0.0],
        "e": [0.0, 2.0, 1.0],
    }
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/all_horizontal_test.py
@pytest.mark.parametrize("expr1", ["a", nw.col("a")])
@pytest.mark.parametrize("expr2", ["b", nw.col("b")])
def test_allh(expr1: Any, expr2: Any) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(duckdb_constructor(data))
    result = df.select(all=nw.all_horizontal(expr1, expr2))

    expected = {"all": [False, False, True]}
    assert_equal_data(result, expected)


def test_allh_all() -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(duckdb_constructor(data))
    result = df.select(all=nw.all_horizontal(nw.all()))
    expected = {"all": [False, False, True]}
    assert_equal_data(result, expected)
    result = df.select(nw.all_horizontal(nw.all()))
    expected = {"a": [False, False, True]}
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/count_test.py
def test_count() -> None:
    data = {"a": [1, 3, 2], "b": [4, None, 6], "z": [7.0, None, None]}
    df = nw.from_native(duckdb_constructor(data))
    result = df.select(nw.col("a", "b", "z").count())
    expected = {"a": [3], "b": [2], "z": [1]}
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/double_test.py
def test_double() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(duckdb_constructor(data))
    result = df.with_columns(nw.all() * 2)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12], "z": [14.0, 16.0, 18.0]}
    assert_equal_data(result, expected)


def test_double_alias() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(duckdb_constructor(data))
    result = df.with_columns(nw.col("a").alias("o"), nw.all() * 2)
    expected = {
        "a": [2, 6, 4],
        "b": [8, 8, 12],
        "z": [14.0, 16.0, 18.0],
        "o": [1, 3, 2],
    }
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/max_test.py
def test_expr_max_expr() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}

    df = nw.from_native(duckdb_constructor(data))
    result = df.select(nw.col("a", "b", "z").max())
    expected = {"a": [3], "b": [6], "z": [9.0]}
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/min_test.py
def test_expr_min_expr() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(duckdb_constructor(data))
    result = df.select(nw.col("a", "b", "z").min())
    expected = {"a": [1], "b": [4], "z": [7.0]}
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/std_test.py
@pytest.mark.xfail
def test_std() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}

    df = nw.from_native(duckdb_constructor(data))
    result = df.select(
        nw.col("a").std().alias("a_ddof_default"),
        nw.col("a").std(ddof=1).alias("a_ddof_1"),
        nw.col("a").std(ddof=0).alias("a_ddof_0"),
        nw.col("b").std(ddof=2).alias("b_ddof_2"),
        nw.col("z").std(ddof=0).alias("z_ddof_0"),
    )
    expected = {
        "a_ddof_default": [1.0],
        "a_ddof_1": [1.0],
        "a_ddof_0": [0.816497],
        "b_ddof_2": [1.632993],
        "z_ddof_0": [0.816497],
    }
    assert_equal_data(result, expected)


# copied from tests/group_by_test.py
def test_group_by_std() -> None:
    data = {"a": [1, 1, 2, 2], "b": [5, 4, 3, 2]}
    result = (
        nw.from_native(duckdb_constructor(data))
        .group_by("a")
        .agg(nw.col("b").std())
        .sort("a")
    )
    expected = {"a": [1, 2], "b": [0.707107] * 2}
    assert_equal_data(result, expected)


def test_group_by_simple_named() -> None:
    data = {"a": [1, 1, 2], "b": [4, 5, 6], "c": [7, 2, 1]}
    df = nw.from_native(duckdb_constructor(data)).lazy()
    result = (
        df.group_by("a")
        .agg(
            b_min=nw.col("b").min(),
            b_max=nw.col("b").max(),
        )
        .sort("a")
        .collect()
    )
    expected = {
        "a": [1, 2],
        "b_min": [4, 6],
        "b_max": [5, 6],
    }
    assert_equal_data(result, expected)


def test_group_by_simple_unnamed() -> None:
    data = {"a": [1, 1, 2], "b": [4, 5, 6], "c": [7, 2, 1]}
    df = nw.from_native(duckdb_constructor(data)).lazy()
    result = (
        df.group_by("a")
        .agg(
            nw.col("b").min(),
            nw.col("c").max(),
        )
        .collect()
        .sort("a")
    )
    expected = {
        "a": [1, 2],
        "b": [4, 6],
        "c": [7, 1],
    }
    assert_equal_data(result, expected)


def test_group_by_multiple_keys() -> None:
    data = {"a": [1, 1, 2], "b": [4, 4, 6], "c": [7, 2, 1]}
    df = nw.from_native(duckdb_constructor(data)).lazy()
    result = (
        df.group_by("a", "b")
        .agg(
            c_min=nw.col("c").min(),
            c_max=nw.col("c").max(),
        )
        .collect()
        .sort("a")
    )
    expected = {
        "a": [1, 2],
        "b": [4, 6],
        "c_min": [2, 1],
        "c_max": [7, 1],
    }
    assert_equal_data(result, expected)
