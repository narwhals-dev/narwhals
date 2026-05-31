from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import DataFrame, assert_equal_data

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [1, 2, 3], "b": ["dogs", "cats", None], "c": ["play", "swim", "walk"]}


XFAIL_TODO = pytest.mark.xfail(
    reason="TODO: Impl `struct(...)`", raises=NotImplementedError
)


@XFAIL_TODO
@pytest.mark.parametrize(
    "exprs",
    [
        (nwp.col("a"), nwp.col("b"), nwp.col("c")),
        ([nwp.col("a"), nwp.col("b"), nwp.col("c")]),
        (nwp.all(),),
        ("a", "b", "c"),
    ],
)
def test_struct_positional_exprs(
    data: Data, dataframe: DataFrame, exprs: tuple[nwp.Expr | list[nwp.Expr], ...]
) -> None:

    df = dataframe(data)
    result = df.select(nwp.struct(*exprs))

    expected = {
        "a": [
            {"a": 1, "b": "dogs", "c": "play"},
            {"a": 2, "b": "cats", "c": "swim"},
            {"a": 3, "b": None, "c": "walk"},
        ]
    }

    assert_equal_data(result, expected)


@XFAIL_TODO
def test_struct_named_exprs(data: Data, dataframe: DataFrame) -> None:

    df = dataframe(data)
    result = df.select(nwp.struct(x="a", y="b").alias("struct"))

    expected = {
        "struct": [{"x": 1, "y": "dogs"}, {"x": 2, "y": "cats"}, {"x": 3, "y": None}]
    }

    assert_equal_data(result, expected)


@XFAIL_TODO
def test_struct_positional_and_named(data: Data, dataframe: DataFrame) -> None:

    df = dataframe(data)
    result = df.select(nwp.struct("a", z="c").alias("struct"))

    expected = {
        "struct": [{"a": 1, "z": "play"}, {"a": 2, "z": "swim"}, {"a": 3, "z": "walk"}]
    }

    assert_equal_data(result, expected)


@XFAIL_TODO
def test_struct_with_expressions(data: Data, dataframe: DataFrame) -> None:

    df = dataframe(data)
    result = df.select(
        nwp.struct(nwp.col("a") * 2, nwp.col("c").str.len_chars()).alias("struct")
    )

    expected = {"struct": [{"a": 2, "c": 4}, {"a": 4, "c": 4}, {"a": 6, "c": 4}]}

    assert_equal_data(result, expected)


@XFAIL_TODO
def test_struct_with_literals(data: Data, dataframe: DataFrame) -> None:

    df = dataframe(data)
    result = df.select(nwp.struct("a", x="c", y=nwp.lit(False)).alias("struct"))

    expected = {
        "struct": [
            {"a": 1, "x": "play", "y": False},
            {"a": 2, "x": "swim", "y": False},
            {"a": 3, "x": "walk", "y": False},
        ]
    }

    assert_equal_data(result, expected)


# TODO @dangotbanned: move to expr_parsing (doesn't require a backend)
@XFAIL_TODO
def test_struct_raise_no_exprs(data: Data, dataframe: DataFrame) -> None:
    df = dataframe(data)
    with pytest.raises(ValueError, match="expected at least 1 expression in 'struct'"):
        df.select(nwp.struct().alias("struct"))

    with pytest.raises(ValueError, match="expected at least 1 expression in 'struct'"):
        df.select(nwp.struct().alias("struct"))


@XFAIL_TODO
def test_struct_with_schema(dataframe: DataFrame) -> None:

    data_numeric = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    schema = {"a": nw.Float64(), "b": nw.Float32()}
    df = dataframe(data_numeric)
    result = df.select(nwp.struct("a", "b").cast(nw.Struct(schema)).alias("struct"))
    assert result.collect_schema()["struct"] == nw.Struct(schema)

    expected = {
        "struct": [{"a": 1.0, "b": 4.0}, {"a": 2.0, "b": 5.0}, {"a": 3.0, "b": 6.0}]
    }
    assert_equal_data(result, expected)


@XFAIL_TODO
def test_struct_with_series(data: Data, dataframe: DataFrame) -> None:

    df = dataframe(data)
    s_a, s_b = df.get_column("a"), df.get_column("b")
    result = df.select(nwp.struct(s_a, s_b).alias("struct"))

    expected = {
        "struct": [{"a": 1, "b": "dogs"}, {"a": 2, "b": "cats"}, {"a": 3, "b": None}]
    }

    assert_equal_data(result, expected)


@XFAIL_TODO
def test_struct_mixed_series_and_exprs(data: Data, dataframe: DataFrame) -> None:

    df = dataframe(data)
    s_a = df.get_column("a")
    result = df.select(nwp.struct(s_a, nwp.col("c")).alias("struct"))

    expected = {
        "struct": [{"a": 1, "c": "play"}, {"a": 2, "c": "swim"}, {"a": 3, "c": "walk"}]
    }

    assert_equal_data(result, expected)


@XFAIL_TODO
def test_struct_named_with_series(data: Data, dataframe: DataFrame) -> None:

    df = dataframe(data)
    s_a = df.get_column("a")
    result = df.select(nwp.struct(x=s_a, y="b").alias("struct"))

    expected = {
        "struct": [{"x": 1, "y": "dogs"}, {"x": 2, "y": "cats"}, {"x": 3, "y": None}]
    }

    assert_equal_data(result, expected)
