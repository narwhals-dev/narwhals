"""
PySpark support in Narwhals is still _very_ limited.
Start with a simple test file whilst we develop the basics.
Once we're a bit further along, we can integrate PySpark tests into the main test suite.
"""

from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from datetime import datetime
from datetime import timezone
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pandas as pd
import pyspark
import pytest

import narwhals.stable.v1 as nw
from narwhals._exceptions import ColumnNotFoundError
from narwhals.utils import parse_version
from tests.utils import compare_dicts

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from narwhals.typing import IntoFrame
    from tests.utils import Constructor


def _pyspark_constructor_with_session(obj: Any, spark_session: SparkSession) -> IntoFrame:
    # NaN and NULL are not the same in PySpark
    pd_df = pd.DataFrame(obj).replace({float("nan"): None})
    return spark_session.createDataFrame(pd_df)  # type: ignore[no-any-return]


@pytest.fixture(params=[_pyspark_constructor_with_session])
def pyspark_constructor(
    request: pytest.FixtureRequest, spark_session: SparkSession
) -> Constructor:
    def _constructor(obj: Any) -> IntoFrame:
        return request.param(obj, spark_session)  # type: ignore[no-any-return]

    return _constructor


# copied from tests/frame/with_columns_test.py
def test_columns(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.columns
    expected = ["a", "b", "z"]
    assert result == expected


# copied from tests/frame/with_columns_test.py
def test_with_columns_order(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.collect_schema().names() == ["a", "b", "z", "d"]
    expected = {"a": [2, 4, 3], "b": [4, 4, 6], "z": [7.0, 8, 9], "d": [0, 2, 1]}
    compare_dicts(result, expected)


@pytest.mark.filterwarnings("ignore:If `index_col` is not specified for `to_spark`")
def test_with_columns_empty(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select().with_columns()
    compare_dicts(result, {})


def test_with_columns_order_single_row(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9], "i": [0, 1, 2]}
    df = nw.from_native(pyspark_constructor(data)).filter(nw.col("i") < 1).drop("i")
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.collect_schema().names() == ["a", "b", "z", "d"]
    expected = {"a": [2], "b": [4], "z": [7.0], "d": [0]}
    compare_dicts(result, expected)


# copied from tests/frame/select_test.py
def test_select(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select("a")
    expected = {"a": [1, 3, 2]}
    compare_dicts(result, expected)


@pytest.mark.filterwarnings("ignore:If `index_col` is not specified for `to_spark`")
def test_empty_select(pyspark_constructor: Constructor) -> None:
    result = nw.from_native(pyspark_constructor({"a": [1, 2, 3]})).lazy().select()
    assert result.collect().shape == (0, 0)


# copied from tests/frame/filter_test.py
def test_filter(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.filter(nw.col("a") > 1)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    compare_dicts(result, expected)


@pytest.mark.filterwarnings("ignore:If `index_col` is not specified for `to_spark`")
def test_filter_with_boolean_list(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))

    context = (
        pytest.raises(
            NotImplementedError,
            match="`LazyFrame.filter` is not supported for PySpark backend with boolean masks.",
        )
        if "pyspark" in str(pyspark_constructor)
        else does_not_raise()
    )

    with context:
        result = df.filter([False, True, True])
        expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
        compare_dicts(result, expected)


# copied from tests/frame/schema_test.py
data = {
    "a": [datetime(2020, 1, 1)],
    "b": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
}


@pytest.mark.filterwarnings("ignore:Determining|Resolving.*")
def test_schema(pyspark_constructor: Constructor) -> None:
    df = nw.from_native(
        pyspark_constructor({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]})
    )
    result = df.schema
    expected = {"a": nw.Int64, "b": nw.Int64, "z": nw.Float64}

    result = df.schema
    assert result == expected
    result = df.lazy().collect().schema
    assert result == expected


def test_collect_schema(pyspark_constructor: Constructor) -> None:
    df = nw.from_native(
        pyspark_constructor({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]})
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
def test_drop(
    pyspark_constructor: Constructor, to_drop: list[str], expected: list[str]
) -> None:
    data = {"abc": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
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
def test_drop_strict(
    pyspark_constructor: Constructor, context: Any, *, strict: bool
) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6]}
    to_drop = ["a", "z"]

    df = nw.from_native(pyspark_constructor(data))

    with context:
        names_out = df.drop(to_drop, strict=strict).collect_schema().names()
        assert names_out == ["b"]


# copied from tests/frame/head_test.py
def test_head(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    expected = {"a": [1, 3], "b": [4, 4], "z": [7.0, 8.0]}

    df_raw = pyspark_constructor(data)
    df = nw.from_native(df_raw)

    result = df.head(2)
    compare_dicts(result, expected)

    result = df.head(2)
    compare_dicts(result, expected)

    # negative indices not allowed for lazyframes
    result = df.lazy().collect().head(-1)
    compare_dicts(result, expected)


# copied from tests/frame/sort_test.py
def test_sort(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.sort("a", "b")
    expected = {
        "a": [1, 2, 3],
        "b": [4, 6, 4],
        "z": [7.0, 9.0, 8.0],
    }
    compare_dicts(result, expected)
    result = df.sort("a", "b", descending=[True, False])
    expected = {
        "a": [3, 2, 1],
        "b": [4, 6, 4],
        "z": [8.0, 9.0, 7.0],
    }
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("nulls_last", "expected"),
    [
        (True, {"a": [0, 2, 0, -1], "b": [3, 2, 1, float("nan")]}),
        (False, {"a": [-1, 0, 2, 0], "b": [float("nan"), 3, 2, 1]}),
    ],
)
def test_sort_nulls(
    pyspark_constructor: Constructor, *, nulls_last: bool, expected: dict[str, float]
) -> None:
    data = {"a": [0, 0, 2, -1], "b": [1, 3, 2, None]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.sort("b", descending=True, nulls_last=nulls_last)
    compare_dicts(result, expected)


# copied from tests/frame/add_test.py
def test_add(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
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
    compare_dicts(result, expected)


# copied from tests/expr_and_series/all_horizontal_test.py
@pytest.mark.parametrize("expr1", ["a", nw.col("a")])
@pytest.mark.parametrize("expr2", ["b", nw.col("b")])
def test_allh(pyspark_constructor: Constructor, expr1: Any, expr2: Any) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(all=nw.all_horizontal(expr1, expr2))

    expected = {"all": [False, False, True]}
    compare_dicts(result, expected)


def test_allh_all(pyspark_constructor: Constructor) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(all=nw.all_horizontal(nw.all()))
    expected = {"all": [False, False, True]}
    compare_dicts(result, expected)
    result = df.select(nw.all_horizontal(nw.all()))
    expected = {"a": [False, False, True]}
    compare_dicts(result, expected)


# copied from tests/expr_and_series/count_test.py
def test_count(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, None, 6], "z": [7.0, None, None]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(nw.col("a", "b", "z").count())
    expected = {"a": [3], "b": [2], "z": [1]}
    compare_dicts(result, expected)


# copied from tests/expr_and_series/double_test.py
def test_double(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.with_columns(nw.all() * 2)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12], "z": [14.0, 16.0, 18.0]}
    compare_dicts(result, expected)


def test_double_alias(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.with_columns(nw.col("a").alias("o"), nw.all() * 2)
    expected = {
        "o": [1, 3, 2],
        "a": [2, 6, 4],
        "b": [8, 8, 12],
        "z": [14.0, 16.0, 18.0],
    }
    compare_dicts(result, expected)


# copied from tests/expr_and_series/max_test.py
def test_expr_max_expr(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}

    df = nw.from_native(pyspark_constructor(data))
    result = df.select(nw.col("a", "b", "z").max())
    expected = {"a": [3], "b": [6], "z": [9.0]}
    compare_dicts(result, expected)


# copied from tests/expr_and_series/min_test.py
def test_expr_min_expr(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(nw.col("a", "b", "z").min())
    expected = {"a": [1], "b": [4], "z": [7.0]}
    compare_dicts(result, expected)


# copied from tests/expr_and_series/std_test.py
def test_std(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}

    df = nw.from_native(pyspark_constructor(data))
    result = df.select(
        nw.col("a").std().alias("a_ddof_default"),
        nw.col("a").std(ddof=1).alias("a_ddof_1"),
        nw.col("a").std(ddof=0).alias("a_ddof_0"),
        nw.col("b").std(ddof=2).alias("b_ddof_2"),
        nw.col("z").std(ddof=0).alias("z_ddof_0"),
    )
    if parse_version(pyspark.__version__) < (3, 4) or parse_version(np.__version__) > (
        2,
        0,
    ):
        expected = {
            "a_ddof_default": [1.0],
            "a_ddof_1": [1.0],
            "a_ddof_0": [1.0],
            "b_ddof_2": [1.154701],
            "z_ddof_0": [1.0],
        }
    else:
        expected = {
            "a_ddof_default": [1.0],
            "a_ddof_1": [1.0],
            "a_ddof_0": [0.816497],
            "b_ddof_2": [1.632993],
            "z_ddof_0": [0.816497],
        }
    compare_dicts(result, expected)


# copied from tests/group_by_test.py
def test_group_by_std(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 1, 2, 2], "b": [5, 4, 3, 2]}
    result = (
        nw.from_native(pyspark_constructor(data))
        .group_by("a")
        .agg(nw.col("b").std())
        .sort("a")
    )
    expected = {"a": [1, 2], "b": [0.707107] * 2}
    compare_dicts(result, expected)


def test_group_by_simple_named(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 1, 2], "b": [4, 5, 6], "c": [7, 2, 1]}
    df = nw.from_native(pyspark_constructor(data)).lazy()
    result = (
        df.group_by("a")
        .agg(
            b_min=nw.col("b").min(),
            b_max=nw.col("b").max(),
        )
        .collect()
        .sort("a")
    )
    expected = {
        "a": [1, 2],
        "b_min": [4, 6],
        "b_max": [5, 6],
    }
    compare_dicts(result, expected)


def test_group_by_simple_unnamed(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 1, 2], "b": [4, 5, 6], "c": [7, 2, 1]}
    df = nw.from_native(pyspark_constructor(data)).lazy()
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
    compare_dicts(result, expected)


def test_group_by_multiple_keys(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 1, 2], "b": [4, 4, 6], "c": [7, 2, 1]}
    df = nw.from_native(pyspark_constructor(data)).lazy()
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
    compare_dicts(result, expected)
