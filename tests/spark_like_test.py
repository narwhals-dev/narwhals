"""PySpark support in Narwhals is still _very_ limited.

Start with a simple test file whilst we develop the basics.
Once we're a bit further along, we can integrate PySpark tests into the main test suite.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING
from typing import Any

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from narwhals.exceptions import ColumnNotFoundError
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from narwhals.typing import IntoFrame
    from tests.utils import Constructor


# Apply filterwarnings to all tests in this module
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:.*is_datetime64tz_dtype is deprecated and will be removed in a future version.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:.*distutils Version classes are deprecated. Use packaging.version instead.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings("ignore: unclosed <socket.socket"),
    pytest.mark.filterwarnings(
        "ignore: The distutils package is deprecated and slated for removal in Python 3.12."
    ),
]


def _pyspark_constructor_with_session(obj: Any, spark_session: SparkSession) -> IntoFrame:
    # NaN and NULL are not the same in PySpark
    pd_df = pd.DataFrame(obj).replace({float("nan"): None}).reset_index()
    return (  # type: ignore[no-any-return]
        spark_session.createDataFrame(pd_df).repartition(2).orderBy("index").drop("index")
    )


@pytest.fixture(params=[_pyspark_constructor_with_session])
def pyspark_constructor(
    request: pytest.FixtureRequest, spark_session: SparkSession
) -> Constructor:
    def _constructor(obj: Any) -> IntoFrame:
        return request.param(obj, spark_session)  # type: ignore[no-any-return]

    return _constructor


# copied from tests/translate/from_native_test.py
def test_series_only(pyspark_constructor: Constructor) -> None:
    obj = pyspark_constructor({"a": [1, 2, 3]})
    with pytest.raises(TypeError, match="Cannot only use `series_only`"):
        _ = nw.from_native(obj, series_only=True)


def test_eager_only_lazy(pyspark_constructor: Constructor) -> None:
    dframe = pyspark_constructor({"a": [1, 2, 3]})
    with pytest.raises(TypeError, match="Cannot only use `eager_only`"):
        _ = nw.from_native(dframe, eager_only=True)


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
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings("ignore:If `index_col` is not specified for `to_spark`")
def test_with_columns_empty(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select().with_columns()
    assert_equal_data(result, {})


def test_with_columns_order_single_row(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9], "i": [0, 1, 2]}
    df = nw.from_native(pyspark_constructor(data)).filter(nw.col("i") < 1).drop("i")
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.collect_schema().names() == ["a", "b", "z", "d"]
    expected = {"a": [2], "b": [4], "z": [7.0], "d": [0]}
    assert_equal_data(result, expected)


# copied from tests/frame/select_test.py
def test_select(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select("a")
    expected = {"a": [1, 3, 2]}
    assert_equal_data(result, expected)


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
    assert_equal_data(result, expected)


# copied from tests/frame/schema_test.py
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
    assert_equal_data(result, expected)

    result = df.head(2)
    assert_equal_data(result, expected)

    # negative indices not allowed for lazyframes
    result = df.lazy().collect().head(-1)
    assert_equal_data(result, expected)


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
        (True, {"a": [0, 2, 0, -1], "b": [3, 2, 1, None]}),
        (False, {"a": [-1, 0, 2, 0], "b": [None, 3, 2, 1]}),
    ],
)
def test_sort_nulls(
    pyspark_constructor: Constructor, *, nulls_last: bool, expected: dict[str, float]
) -> None:
    data = {"a": [0, 0, 2, -1], "b": [1, 3, 2, None]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.sort("b", descending=True, nulls_last=nulls_last).lazy().collect()
    assert_equal_data(result, expected)


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
    assert_equal_data(result, expected)


def test_abs(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 2, 3, -4, 5]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(nw.col("a").abs())
    expected = {"a": [1, 2, 3, 4, 5]}
    assert_equal_data(result, expected)


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
    assert_equal_data(result, expected)


def test_allh_all(pyspark_constructor: Constructor) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(all=nw.all_horizontal(nw.all()))
    expected = {"all": [False, False, True]}
    assert_equal_data(result, expected)
    result = df.select(nw.all_horizontal(nw.all()))
    expected = {"a": [False, False, True]}
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/sum_horizontal_test.py
@pytest.mark.parametrize("col_expr", [nw.col("a"), "a"])
def test_sumh(pyspark_constructor: Constructor, col_expr: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.with_columns(horizontal_sum=nw.sum_horizontal(col_expr, nw.col("b")))
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "horizontal_sum": [5, 7, 8],
    }
    assert_equal_data(result, expected)


def test_sumh_nullable(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 8, 3], "b": [4, 5, None], "idx": [0, 1, 2]}
    expected = {"hsum": [5, 13, 3]}

    df = nw.from_native(pyspark_constructor(data))
    result = df.select("idx", hsum=nw.sum_horizontal("a", "b")).sort("idx").drop("idx")
    assert_equal_data(result, expected)


def test_sumh_all(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 2, 3], "b": [10, 20, 30]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(nw.sum_horizontal(nw.all()))
    expected = {
        "a": [11, 22, 33],
    }
    assert_equal_data(result, expected)
    result = df.select(c=nw.sum_horizontal(nw.all()))
    expected = {
        "c": [11, 22, 33],
    }
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/count_test.py
def test_count(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 2, 3], "b": [4, None, 6], "z": [7.0, None, None]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(nw.col("a", "b", "z").count())
    expected = {"a": [3], "b": [2], "z": [1]}
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/double_test.py
def test_double(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.with_columns(nw.all() * 2)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12], "z": [14.0, 16.0, 18.0]}
    assert_equal_data(result, expected)


def test_double_alias(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.with_columns(nw.col("a").alias("o"), nw.all() * 2)
    expected = {
        "a": [2, 6, 4],
        "b": [8, 8, 12],
        "z": [14.0, 16.0, 18.0],
        "o": [1, 3, 2],
    }
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/max_test.py
def test_expr_max_expr(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}

    df = nw.from_native(pyspark_constructor(data))
    result = df.select(nw.col("a", "b", "z").max())
    expected = {"a": [3], "b": [6], "z": [9.0]}
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/min_test.py
def test_expr_min_expr(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(nw.col("a", "b", "z").min())
    expected = {"a": [1], "b": [4], "z": [7.0]}
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/min_test.py
@pytest.mark.parametrize("expr", [nw.col("a", "b", "z").sum(), nw.sum("a", "b", "z")])
def test_expr_sum_expr(pyspark_constructor: Constructor, expr: nw.Expr) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(expr)
    expected = {"a": [6], "b": [14], "z": [24.0]}
    assert_equal_data(result, expected)


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
    expected = {
        "a_ddof_default": [1.0],
        "a_ddof_1": [1.0],
        "a_ddof_0": [0.816497],
        "b_ddof_2": [1.632993],
        "z_ddof_0": [0.816497],
    }
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/var_test.py
def test_var(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2, None], "b": [4, 4, 6, None], "z": [7.0, 8, 9, None]}

    expected_results = {
        "a_ddof_1": [1.0],
        "a_ddof_0": [0.6666666666666666],
        "b_ddof_2": [2.666666666666667],
        "z_ddof_0": [0.6666666666666666],
    }

    df = nw.from_native(pyspark_constructor(data))
    result = df.select(
        nw.col("a").var(ddof=1).alias("a_ddof_1"),
        nw.col("a").var(ddof=0).alias("a_ddof_0"),
        nw.col("b").var(ddof=2).alias("b_ddof_2"),
        nw.col("z").var(ddof=0).alias("z_ddof_0"),
    )
    assert_equal_data(result, expected_results)


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
    assert_equal_data(result, expected)


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
    assert_equal_data(result, expected)


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
    assert_equal_data(result, expected)


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
    assert_equal_data(result, expected)


# copied from tests/group_by_test.py
@pytest.mark.parametrize(
    ("attr", "ddof"),
    [
        ("std", 0),
        ("var", 0),
        ("std", 2),
        ("var", 2),
    ],
)
def test_group_by_depth_1_std_var(
    pyspark_constructor: Constructor,
    attr: str,
    ddof: int,
) -> None:
    data = {"a": [1, 1, 1, 2, 2, 2], "b": [4, 5, 6, 0, 5, 5]}
    _pow = 0.5 if attr == "std" else 1
    expected = {
        "a": [1, 2],
        "b": [
            (sum((v - 5) ** 2 for v in [4, 5, 6]) / (3 - ddof)) ** _pow,
            (sum((v - 10 / 3) ** 2 for v in [0, 5, 5]) / (3 - ddof)) ** _pow,
        ],
    }
    expr = getattr(nw.col("b"), attr)(ddof=ddof)
    result = nw.from_native(pyspark_constructor(data)).group_by("a").agg(expr).sort("a")
    assert_equal_data(result, expected)


# copied from tests/frame/drop_nulls_test.py
def test_drop_nulls(pyspark_constructor: Constructor) -> None:
    data = {
        "a": [1.0, 2.0, None, 4.0],
        "b": [None, 3.0, None, 5.0],
    }

    result = nw.from_native(pyspark_constructor(data)).drop_nulls()
    expected = {
        "a": [2.0, 4.0],
        "b": [3.0, 5.0],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("a", {"a": [1, 2.0, 4.0], "b": [None, 3.0, 5.0]}),
        (["a"], {"a": [1, 2.0, 4.0], "b": [None, 3.0, 5.0]}),
        (["a", "b"], {"a": [2.0, 4.0], "b": [3.0, 5.0]}),
    ],
)
def test_drop_nulls_subset(
    pyspark_constructor: Constructor,
    subset: str | list[str],
    expected: dict[str, float],
) -> None:
    data = {
        "a": [1.0, 2.0, None, 4.0],
        "b": [None, 3.0, None, 5.0],
    }

    result = nw.from_native(pyspark_constructor(data)).drop_nulls(subset=subset)
    assert_equal_data(result, expected)


# copied from tests/frame/rename_test.py
def test_rename(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.rename({"a": "x", "b": "y"})
    expected = {"x": [1, 3, 2], "y": [4, 4, 6], "z": [7.0, 8, 9]}
    assert_equal_data(result, expected)


# adapted from tests/frame/unique_test.py
@pytest.mark.parametrize("subset", ["b", ["b"]])
@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("first", {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}),
        ("last", {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}),
        ("any", {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}),
        ("none", {"a": [2], "b": [6], "z": [9]}),
    ],
)
def test_unique(
    pyspark_constructor: Constructor,
    subset: str | list[str] | None,
    keep: str,
    expected: dict[str, list[float]],
) -> None:
    if keep == "any":
        context: Any = does_not_raise()
    elif keep == "none":
        context = pytest.raises(
            ValueError,
            match=r"`LazyFrame.unique` with PySpark backend only supports `keep='any'`.",
        )
    else:
        context = pytest.raises(ValueError, match=f": {keep}")

    with context:
        data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
        df = nw.from_native(pyspark_constructor(data))

        result = df.unique(subset, keep=keep).sort("z")  # type: ignore[arg-type]
        assert_equal_data(result, expected)


def test_unique_none(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.unique().sort("z")
    assert_equal_data(result, data)


def test_inner_join_two_keys(pyspark_constructor: Constructor) -> None:
    data = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zorro": [7.0, 8, 9],
        "idx": [0, 1, 2],
    }
    df = nw.from_native(pyspark_constructor(data))
    df_right = nw.from_native(pyspark_constructor(data))
    result = df.join(
        df_right,  # type: ignore[arg-type]
        left_on=["antananarivo", "bob"],
        right_on=["antananarivo", "bob"],
        how="inner",
    )
    result = result.sort("idx").drop("idx_right")

    df = nw.from_native(pyspark_constructor(data))
    df_right = nw.from_native(pyspark_constructor(data))

    result_on = df.join(df_right, on=["antananarivo", "bob"], how="inner")  # type: ignore[arg-type]
    result_on = result_on.sort("idx").drop("idx_right")
    expected = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zorro": [7.0, 8, 9],
        "idx": [0, 1, 2],
        "zorro_right": [7.0, 8, 9],
    }
    assert_equal_data(result, expected)
    assert_equal_data(result_on, expected)


def test_inner_join_single_key(pyspark_constructor: Constructor) -> None:
    data = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zorro": [7.0, 8, 9],
        "idx": [0, 1, 2],
    }
    df = nw.from_native(pyspark_constructor(data))
    df_right = nw.from_native(pyspark_constructor(data))
    result = (
        df.join(
            df_right,  # type: ignore[arg-type]
            left_on="antananarivo",
            right_on="antananarivo",
            how="inner",
        )
        .sort("idx")
        .drop("idx_right")
    )

    df = nw.from_native(pyspark_constructor(data))
    df_right = nw.from_native(pyspark_constructor(data))
    result_on = (
        df.join(
            df_right,  # type: ignore[arg-type]
            on="antananarivo",
            how="inner",
        )
        .sort("idx")
        .drop("idx_right")
    )

    expected = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zorro": [7.0, 8, 9],
        "idx": [0, 1, 2],
        "bob_right": [4, 4, 6],
        "zorro_right": [7.0, 8, 9],
    }
    assert_equal_data(result, expected)
    assert_equal_data(result_on, expected)


def test_cross_join(pyspark_constructor: Constructor) -> None:
    data = {"antananarivo": [1, 3, 2]}
    df = nw.from_native(pyspark_constructor(data))
    other = nw.from_native(pyspark_constructor(data))
    result = df.join(other, how="cross").sort("antananarivo", "antananarivo_right")  # type: ignore[arg-type]
    expected = {
        "antananarivo": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "antananarivo_right": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
    assert_equal_data(result, expected)

    with pytest.raises(
        ValueError,
        match="Can not pass `left_on`, `right_on` or `on` keys for cross join",
    ):
        df.join(other, how="cross", left_on="antananarivo")  # type: ignore[arg-type]


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("suffix", ["_right", "_custom_suffix"])
def test_suffix(pyspark_constructor: Constructor, how: str, suffix: str) -> None:
    data = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zorro": [7.0, 8, 9],
    }
    df = nw.from_native(pyspark_constructor(data))
    df_right = nw.from_native(pyspark_constructor(data))
    result = df.join(
        df_right,  # type: ignore[arg-type]
        left_on=["antananarivo", "bob"],
        right_on=["antananarivo", "bob"],
        how=how,  # type: ignore[arg-type]
        suffix=suffix,
    )
    result_cols = result.collect_schema().names()
    assert result_cols == ["antananarivo", "bob", "zorro", f"zorro{suffix}"]


@pytest.mark.parametrize("suffix", ["_right", "_custom_suffix"])
def test_cross_join_suffix(pyspark_constructor: Constructor, suffix: str) -> None:
    data = {"antananarivo": [1, 3, 2]}
    df = nw.from_native(pyspark_constructor(data))
    other = nw.from_native(pyspark_constructor(data))
    result = df.join(other, how="cross", suffix=suffix).sort(  # type: ignore[arg-type]
        "antananarivo", f"antananarivo{suffix}"
    )
    expected = {
        "antananarivo": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        f"antananarivo{suffix}": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("join_key", "filter_expr", "expected"),
    [
        (
            ["antananarivo", "bob"],
            (nw.col("bob") < 5),
            {"antananarivo": [2], "bob": [6], "zorro": [9]},
        ),
        (["bob"], (nw.col("bob") < 5), {"antananarivo": [2], "bob": [6], "zorro": [9]}),
        (
            ["bob"],
            (nw.col("bob") > 5),
            {"antananarivo": [1, 3], "bob": [4, 4], "zorro": [7.0, 8.0]},
        ),
    ],
)
def test_anti_join(
    pyspark_constructor: Constructor,
    join_key: list[str],
    filter_expr: nw.Expr,
    expected: dict[str, list[Any]],
) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zorro": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    other = df.filter(filter_expr)
    result = df.join(other, how="anti", left_on=join_key, right_on=join_key)  # type: ignore[arg-type]
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("join_key", "filter_expr", "expected"),
    [
        (
            "antananarivo",
            (nw.col("bob") > 5),
            {"antananarivo": [2], "bob": [6], "zorro": [9]},
        ),
        (
            ["antananarivo"],
            (nw.col("bob") > 5),
            {"antananarivo": [2], "bob": [6], "zorro": [9]},
        ),
        (
            ["bob"],
            (nw.col("bob") < 5),
            {"antananarivo": [1, 3], "bob": [4, 4], "zorro": [7, 8]},
        ),
        (
            ["antananarivo", "bob"],
            (nw.col("bob") < 5),
            {"antananarivo": [1, 3], "bob": [4, 4], "zorro": [7, 8]},
        ),
    ],
)
def test_semi_join(
    pyspark_constructor: Constructor,
    join_key: list[str],
    filter_expr: nw.Expr,
    expected: dict[str, list[Any]],
) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zorro": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data))
    other = df.filter(filter_expr)
    result = df.join(other, how="semi", left_on=join_key, right_on=join_key).sort(  # type: ignore[arg-type]
        "antananarivo"
    )
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings("ignore:the default coalesce behavior")
def test_left_join(pyspark_constructor: Constructor) -> None:
    data_left = {
        "antananarivo": [1.0, 2, 3],
        "bob": [4.0, 5, 6],
        "idx": [0.0, 1.0, 2.0],
    }
    data_right = {
        "antananarivo": [1.0, 2, 3],
        "co": [4.0, 5, 7],
        "idx": [0.0, 1.0, 2.0],
    }
    df_left = nw.from_native(pyspark_constructor(data_left))
    df_right = nw.from_native(pyspark_constructor(data_right))
    result = (
        df_left.join(df_right, left_on="bob", right_on="co", how="left")  # type: ignore[arg-type]
        .sort("idx")
        .drop("idx_right")
    )
    expected = {
        "antananarivo": [1, 2, 3],
        "bob": [4, 5, 6],
        "idx": [0, 1, 2],
        "antananarivo_right": [1, 2, None],
    }
    assert_equal_data(result, expected)

    df_left = nw.from_native(pyspark_constructor(data_left))
    df_right = nw.from_native(pyspark_constructor(data_right))
    result_on_list = df_left.join(
        df_right,  # type: ignore[arg-type]
        on=["antananarivo", "idx"],
        how="left",
    )
    result_on_list = result_on_list.sort("idx")
    expected_on_list = {
        "antananarivo": [1, 2, 3],
        "bob": [4, 5, 6],
        "idx": [0, 1, 2],
        "co": [4, 5, 7],
    }
    assert_equal_data(result_on_list, expected_on_list)


@pytest.mark.filterwarnings("ignore: the default coalesce behavior")
def test_left_join_multiple_column(pyspark_constructor: Constructor) -> None:
    data_left = {"antananarivo": [1, 2, 3], "bob": [4, 5, 6], "idx": [0, 1, 2]}
    data_right = {"antananarivo": [1, 2, 3], "c": [4, 5, 6], "idx": [0, 1, 2]}
    df_left = nw.from_native(pyspark_constructor(data_left))
    df_right = nw.from_native(pyspark_constructor(data_right))
    result = (
        df_left.join(
            df_right,  # type: ignore[arg-type]
            left_on=["antananarivo", "bob"],
            right_on=["antananarivo", "c"],
            how="left",
        )
        .sort("idx")
        .drop("idx_right")
    )
    expected = {"antananarivo": [1, 2, 3], "bob": [4, 5, 6], "idx": [0, 1, 2]}
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings("ignore: the default coalesce behavior")
def test_left_join_overlapping_column(pyspark_constructor: Constructor) -> None:
    data_left = {
        "antananarivo": [1.0, 2, 3],
        "bob": [4.0, 5, 6],
        "d": [1.0, 4, 2],
        "idx": [0.0, 1.0, 2.0],
    }
    data_right = {
        "antananarivo": [1.0, 2, 3],
        "c": [4.0, 5, 6],
        "d": [1.0, 4, 2],
        "idx": [0.0, 1.0, 2.0],
    }
    df_left = nw.from_native(pyspark_constructor(data_left))
    df_right = nw.from_native(pyspark_constructor(data_right))
    result = df_left.join(df_right, left_on="bob", right_on="c", how="left").sort("idx")  # type: ignore[arg-type]
    result = result.drop("idx_right")
    expected: dict[str, list[Any]] = {
        "antananarivo": [1, 2, 3],
        "bob": [4, 5, 6],
        "d": [1, 4, 2],
        "idx": [0, 1, 2],
        "antananarivo_right": [1, 2, 3],
        "d_right": [1, 4, 2],
    }
    assert_equal_data(result, expected)

    df_left = nw.from_native(pyspark_constructor(data_left))
    df_right = nw.from_native(pyspark_constructor(data_right))
    result = (
        df_left.join(
            df_right,  # type: ignore[arg-type]
            left_on="antananarivo",
            right_on="d",
            how="left",
        )
        .sort("idx")
        .drop("idx_right")
    )
    expected = {
        "antananarivo": [1, 2, 3],
        "bob": [4, 5, 6],
        "d": [1, 4, 2],
        "idx": [0, 1, 2],
        "antananarivo_right": [1.0, 3.0, None],
        "c": [4.0, 6.0, None],
    }
    assert_equal_data(result, expected)


@pytest.mark.xfail(
    sys.version_info < (3, 9),
    reason="median() not supported on Python 3.8",
)
def test_median(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2, None, float("nan")]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(median=nw.col("a").median())
    expected = {"median": [2.0]}
    assert_equal_data(result, expected)


# copied from tests/expr_and_series/clip_test.py
def test_clip(pyspark_constructor: Constructor) -> None:
    df = nw.from_native(pyspark_constructor({"a": [1, 2, 3, -4, 5]}))
    result = df.select(
        lower_only=nw.col("a").clip(lower_bound=3),
        upper_only=nw.col("a").clip(upper_bound=4),
        both=nw.col("a").clip(3, 4),
    )
    expected = {
        "lower_only": [3, 3, 3, 3, 5],
        "upper_only": [1, 2, 3, -4, 4],
        "both": [3, 3, 3, 3, 4],
    }
    assert_equal_data(result, expected)


def test_is_between(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2, 5, 4]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(
        both=nw.col("a").is_between(2, 4, closed="both"),
        neither=nw.col("a").is_between(2, 4, closed="neither"),
        left=nw.col("a").is_between(2, 4, closed="left"),
        right=nw.col("a").is_between(2, 4, closed="right"),
    )
    expected = {
        "both": [False, True, True, False, True],
        "neither": [False, True, False, False, False],
        "left": [False, True, True, False, False],
        "right": [False, True, False, False, True],
    }
    assert_equal_data(result, expected)


def test_is_duplicated(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 2, 2, 3, 4, 4]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(duplicated=nw.col("a").is_duplicated())
    expected = {"duplicated": [False, True, True, False, True, True]}
    assert_equal_data(result, expected)


def test_is_nan(pyspark_constructor: Constructor) -> None:
    data = {"a": [1.0, float("nan"), 2.0, None, 3.0]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(nan=nw.col("a").is_nan())
    expected = {"nan": [False, True, False, True, False]}
    assert_equal_data(result, expected)


def test_is_finite(pyspark_constructor: Constructor) -> None:
    data = {"a": [1.0, float("inf"), float("-inf"), None, 2.0]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(finite=nw.col("a").is_finite())
    expected = {"finite": [True, False, False, False, True]}
    assert_equal_data(result, expected)


def test_is_in(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 2, 3, 4, 5]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(in_list=nw.col("a").is_in([2, 4]))
    expected = {"in_list": [False, True, False, True, False]}
    assert_equal_data(result, expected)


def test_is_unique(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 2, 2, 3, 4, 4]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(unique=nw.col("a").is_unique())
    expected = {"unique": [True, False, False, True, False, False]}
    assert_equal_data(result, expected)


def test_len(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 2, 3, 4, 5]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(length=nw.col("a").len())
    expected = {"length": [5]}
    assert_equal_data(result, expected)


def test_n_unique(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 2, 2, 3, 4, 4]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(n_unique=nw.col("a").n_unique())
    expected = {"n_unique": [4]}
    assert_equal_data(result, expected)


# Copied from tests/expr_and_series/round_test.py
@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_round(pyspark_constructor: Constructor, decimals: int) -> None:
    data = {"a": [2.12345, 2.56789, 3.901234]}
    df = nw.from_native(pyspark_constructor(data))

    expected_data = {k: [round(e, decimals) for e in v] for k, v in data.items()}
    result_frame = df.select(nw.col("a").round(decimals))
    assert_equal_data(result_frame, expected_data)


def test_skew(pyspark_constructor: Constructor) -> None:
    data = {"a": [1, 2, 3, 2, 1]}
    df = nw.from_native(pyspark_constructor(data))
    result = df.select(skew=nw.col("a").skew())
    expected = {"skew": [0.343622]}
    assert_equal_data(result, expected)
