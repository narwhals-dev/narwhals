from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import Implementation
from tests.utils import compare_dicts


def test_inner_join_two_keys(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9], "index": [0, 1, 2]}
    df = nw.from_native(constructor(data))
    df_right = df
    result = df.join(df_right, left_on=["a", "b"], right_on=["a", "b"], how="inner")  # type: ignore[arg-type]
    result = result.sort("index")
    result = result.drop("index_right")
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8, 9],
        "z_right": [7.0, 8, 9],
        "index": [0, 1, 2],
    }
    compare_dicts(result, expected)


def test_inner_join_single_key(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9], "index": [0, 1, 2]}
    df = nw.from_native(constructor(data))
    df_right = df
    result = df.join(df_right, left_on="a", right_on="a", how="inner").sort("index")  # type: ignore[arg-type]
    result = result.drop("index_right")
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "b_right": [4, 4, 6],
        "z": [7.0, 8, 9],
        "z_right": [7.0, 8, 9],
        "index": [0, 1, 2],
    }
    compare_dicts(result, expected)


def test_cross_join(constructor: Any) -> None:
    data = {"a": [1, 3, 2]}
    df = nw.from_native(constructor(data))
    result = df.join(df, how="cross").sort("a", "a_right")  # type: ignore[arg-type]
    expected = {
        "a": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "a_right": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
    compare_dicts(result, expected)

    with pytest.raises(ValueError, match="Can not pass left_on, right_on for cross join"):
        df.join(df, how="cross", left_on="a")  # type: ignore[arg-type]


def test_cross_join_non_pandas() -> None:
    data = {"a": [1, 3, 2]}
    df = nw.from_native(pd.DataFrame(data))
    # HACK to force testing for a non-pandas codepath
    df._compliant_frame._implementation = Implementation.MODIN
    result = df.join(df, how="cross")  # type: ignore[arg-type]
    expected = {
        "a": [1, 1, 1, 3, 3, 3, 2, 2, 2],
        "a_right": [1, 3, 2, 1, 3, 2, 1, 3, 2],
    }
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("join_key", "filter_expr", "expected"),
    [
        (["a", "b"], (nw.col("b") < 5), {"a": [2], "b": [6], "z": [9]}),
        (["b"], (nw.col("b") < 5), {"a": [2], "b": [6], "z": [9]}),
        (["b"], (nw.col("b") > 5), {"a": [1, 3], "b": [4, 4], "z": [7.0, 8.0]}),
    ],
)
def test_anti_join(
    constructor: Any,
    join_key: list[str],
    filter_expr: nw.Expr,
    expected: dict[str, list[Any]],
) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    other = df.filter(filter_expr)
    result = df.join(other, how="anti", left_on=join_key, right_on=join_key)  # type: ignore[arg-type]
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("join_key", "filter_expr", "expected"),
    [
        (["a"], (nw.col("b") > 5), {"a": [2], "b": [6], "z": [9]}),
        (["b"], (nw.col("b") < 5), {"a": [1, 3], "b": [4, 4], "z": [7, 8]}),
        (["a", "b"], (nw.col("b") < 5), {"a": [1, 3], "b": [4, 4], "z": [7, 8]}),
    ],
)
def test_semi_join(
    constructor: Any,
    join_key: list[str],
    filter_expr: nw.Expr,
    expected: dict[str, list[Any]],
) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    other = df.filter(filter_expr)
    result = df.join(other, how="semi", left_on=join_key, right_on=join_key).sort("a")  # type: ignore[arg-type]
    compare_dicts(result, expected)


@pytest.mark.parametrize("how", ["right", "full"])
def test_join_not_implemented(constructor: Any, how: str) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            f"Only the following join strategies are supported: ('inner', 'left', 'cross', 'anti', 'semi'); found '{how}'."
        ),
    ):
        df.join(df, left_on="a", right_on="a", how=how)  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore:the default coalesce behavior")
def test_left_join(constructor: Any) -> None:
    data_left = {"a": [1.0, 2, 3], "b": [4.0, 5, 6], "index": [0.0, 1.0, 2.0]}
    data_right = {"a": [1.0, 2, 3], "c": [4.0, 5, 7], "index": [0.0, 1.0, 2.0]}
    df_left = nw.from_native(constructor(data_left))
    df_right = nw.from_native(constructor(data_right))
    result = df_left.join(df_right, left_on="b", right_on="c", how="left").select(  # type: ignore[arg-type]
        nw.all().fill_null(float("nan"))
    )
    result = result.sort("index")
    result = result.drop("index_right")
    expected = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "a_right": [1, 2, float("nan")],
        "index": [0, 1, 2],
    }
    compare_dicts(result, expected)


@pytest.mark.filterwarnings("ignore: the default coalesce behavior")
def test_left_join_multiple_column(constructor: Any) -> None:
    data_left = {"a": [1, 2, 3], "b": [4, 5, 6], "index": [0, 1, 2]}
    data_right = {"a": [1, 2, 3], "c": [4, 5, 6], "index": [0, 1, 2]}
    df_left = nw.from_native(constructor(data_left))
    df_right = nw.from_native(constructor(data_right))
    result = df_left.join(df_right, left_on=["a", "b"], right_on=["a", "c"], how="left")  # type: ignore[arg-type]
    result = result.sort("index")
    result = result.drop("index_right")
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "index": [0, 1, 2]}
    compare_dicts(result, expected)


@pytest.mark.filterwarnings("ignore: the default coalesce behavior")
def test_left_join_overlapping_column(constructor: Any) -> None:
    data_left = {
        "a": [1.0, 2, 3],
        "b": [4.0, 5, 6],
        "d": [1.0, 4, 2],
        "index": [0.0, 1.0, 2.0],
    }
    data_right = {
        "a": [1.0, 2, 3],
        "c": [4.0, 5, 6],
        "d": [1.0, 4, 2],
        "index": [0.0, 1.0, 2.0],
    }
    df_left = nw.from_native(constructor(data_left))
    df_right = nw.from_native(constructor(data_right))
    result = df_left.join(df_right, left_on="b", right_on="c", how="left").sort("index")  # type: ignore[arg-type]
    result = result.drop("index_right")
    expected: dict[str, list[Any]] = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "d": [1, 4, 2],
        "a_right": [1, 2, 3],
        "d_right": [1, 4, 2],
        "index": [0, 1, 2],
    }
    compare_dicts(result, expected)
    result = df_left.join(df_right, left_on="a", right_on="d", how="left").select(  # type: ignore[arg-type]
        nw.all().fill_null(float("nan"))
    )
    result = result.sort("index")
    result = result.drop("index_right")
    expected = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "d": [1, 4, 2],
        "a_right": [1.0, 3.0, float("nan")],
        "c": [4.0, 6.0, float("nan")],
        "index": [0, 1, 2],
    }
    compare_dicts(result, expected)


def test_joinasof_numeric(constructor: Any, request: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 5, 10], "val": ["a", "b", "c"]}))
    df_right = nw.from_native(constructor({"a": [1, 2, 3, 6, 7], "val": [1, 2, 3, 6, 7]}))
    result_backward = df.join_asof(df_right, left_on="a", right_on="a")  # type: ignore[arg-type]
    result_forward = df.join_asof(df_right, left_on="a", right_on="a", strategy="forward")  # type: ignore[arg-type]
    result_nearest = df.join_asof(df_right, left_on="a", right_on="a", strategy="nearest")  # type: ignore[arg-type]
    expected_backward = {
        "a": [1, 5, 10],
        "val": ["a", "b", "c"],
        "val_right": [1, 3, 7],
    }
    expected_forward = {
        "a": [1, 5, 10],
        "val": ["a", "b", "c"],
        "val_right": [1, 6, float("nan")],
    }
    expected_nearest = {
        "a": [1, 5, 10],
        "val": ["a", "b", "c"],
        "val_right": [1, 6, 7],
    }
    compare_dicts(result_backward, expected_backward)
    compare_dicts(result_forward, expected_forward)
    compare_dicts(result_nearest, expected_nearest)


def test_joinasof_time(constructor: Any, request: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(
        constructor(
            {
                "datetime": [
                    datetime(2016, 3, 1),
                    datetime(2018, 8, 1),
                    datetime(2019, 1, 1),
                ],
                "population": [82.19, 82.66, 83.12],
            }
        )
    )
    df_right = nw.from_native(
        constructor(
            {
                "datetime": [
                    datetime(2016, 1, 1),
                    datetime(2017, 1, 1),
                    datetime(2018, 1, 1),
                    datetime(2019, 1, 1),
                    datetime(2020, 1, 1),
                ],
                "gdp": [4164, 4411, 4566, 4696, 4827],
            }
        )
    )
    result_backward = df.join_asof(df_right, left_on="datetime", right_on="datetime")  # type: ignore[arg-type]
    result_forward = df.join_asof(
        df_right,  # type: ignore[arg-type]
        left_on="datetime",
        right_on="datetime",
        strategy="forward",
    )
    result_nearest = df.join_asof(
        df_right,  # type: ignore[arg-type]
        left_on="datetime",
        right_on="datetime",
        strategy="nearest",
    )
    expected_backward = {
        "datetime": [datetime(2016, 3, 1), datetime(2018, 8, 1), datetime(2019, 1, 1)],
        "population": [82.19, 82.66, 83.12],
        "gdp": [4164, 4566, 4696],
    }
    expected_forward = {
        "datetime": [datetime(2016, 3, 1), datetime(2018, 8, 1), datetime(2019, 1, 1)],
        "population": [82.19, 82.66, 83.12],
        "gdp": [4411, 4696, 4696],
    }
    expected_nearest = {
        "datetime": [datetime(2016, 3, 1), datetime(2018, 8, 1), datetime(2019, 1, 1)],
        "population": [82.19, 82.66, 83.12],
        "gdp": [4164, 4696, 4696],
    }
    compare_dicts(result_backward, expected_backward)
    compare_dicts(result_forward, expected_forward)
    compare_dicts(result_nearest, expected_nearest)
