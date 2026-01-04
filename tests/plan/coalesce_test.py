from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data_int() -> Data:
    return {
        "a": [0, None, None, None, None],
        "b": [1, None, None, 5, 3],
        "c": [5, None, 3, 2, 1],
    }


@pytest.fixture(scope="module")
def data_str() -> Data:
    return {
        "a": ["0", None, None, None, None],
        "b": ["1", None, None, "5", "3"],
        "c": ["5", None, "3", "2", "1"],
    }


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwp.coalesce("a", "b", "c"), {"a": [0, None, 3, 5, 3]}),
        (
            nwp.coalesce("a", "b", "c", nwp.lit(-100)).alias("lit"),
            {"lit": [0, -100, 3, 5, 3]},
        ),
        (
            nwp.coalesce(nwp.lit(None, nw.Int64), "b", "c", 500).alias("into_lit"),
            {"into_lit": [1, 500, 3, 5, 3]},
        ),
    ],
)
def test_coalesce_numeric(data_int: Data, expr: nwp.Expr, expected: Data) -> None:
    result = dataframe(data_int).select(expr)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            nwp.coalesce("a", "b", "c").alias("no_lit"),
            {"no_lit": ["0", None, "3", "5", "3"]},
        ),
        (nwp.coalesce("a", "b", "c", nwp.lit("xyz")), {"a": ["0", "xyz", "3", "5", "3"]}),
    ],
)
def test_coalesce_strings(data_str: Data, expr: nwp.Expr, expected: Data) -> None:
    result = dataframe(data_str).select(expr)
    assert_equal_data(result, expected)


def test_coalesce_series(data_str: Data) -> None:
    df = dataframe(data_str)
    ser = df.get_column("b").alias("b_renamed")
    exprs = nwp.coalesce(ser, "a", nwp.col("c").fill_null("filled")), nwp.lit("ignored")
    result = df.select(exprs)
    assert_equal_data(result, {"b_renamed": ["1", "filled", "3", "5", "3"]})


def test_coalesce_raises_non_expr() -> None:
    class NotAnExpr: ...

    with pytest.raises(
        TypeError, match=re.escape("'NotAnExpr' is not supported in `nw.lit`")
    ):
        nwp.coalesce("a", "b", "c", NotAnExpr())  # type: ignore[arg-type]


def test_coalesce_multi_output() -> None:
    data = {
        "col1": [True, None, False, False, None],
        "col2": [True, False, True, False, None],
    }
    df = dataframe(data)
    result = df.select(nwp.coalesce(nwp.all(), True))
    expected = {"col1": [True, False, False, False, True]}
    assert_equal_data(result, expected)
