from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from tests.plan.utils import DataFrame, assert_equal_data

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "a": [False, False, True],
        "b": [False, True, True],
        "c": [True, True, False],
        "d": [True, None, None],
        "e": [None, True, False],
    }


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            nwp.all_horizontal("a", nwp.col("b"), ignore_nulls=True),
            {"a": [False, False, True]},
        ),
        pytest.param(
            nwp.all_horizontal("c", "d", ignore_nulls=True),
            {"c": [True, True, False]},
            id="ignore_nulls-1",
        ),
        (nwp.all_horizontal("c", "d", ignore_nulls=False), {"c": [True, None, False]}),
        (
            nwp.all_horizontal(nwp.nth(0, 1), ignore_nulls=True),
            {"a": [False, False, True]},
        ),
        pytest.param(
            nwp.all_horizontal(
                nwp.col("a"), nwp.nth(0), ncs.first(), "a", ignore_nulls=True
            ),
            {"a": [False, False, True]},
            id="duplicated",
        ),
        (
            nwp.all_horizontal(nwp.exclude("a", "b"), ignore_nulls=False),
            {"c": [None, None, False]},
        ),
        pytest.param(
            nwp.all_horizontal(ncs.all() - ncs.by_index(0, 1), ignore_nulls=True),
            {"c": [True, True, False]},
            id="ignore_nulls-2",
        ),
    ],
)
def test_all_horizontal(
    data: Data, expr: nwp.Expr, expected: Data, dataframe: DataFrame
) -> None:
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            nwp.any_horizontal("a", nwp.col("b"), ignore_nulls=True),
            {"a": [False, True, True]},
        ),
        (nwp.any_horizontal("c", "d", ignore_nulls=False), {"c": [True, True, None]}),
        pytest.param(
            nwp.any_horizontal("c", "d", ignore_nulls=True),
            {"c": [True, True, False]},
            id="ignore_nulls-1",
        ),
        (
            nwp.any_horizontal(nwp.nth(0, 1), ignore_nulls=False),
            {"a": [False, True, True]},
        ),
        pytest.param(
            nwp.any_horizontal(
                nwp.col("a"), nwp.nth(0), ncs.first(), "a", ignore_nulls=True
            ),
            {"a": [False, False, True]},
            id="duplicated",
        ),
        (
            nwp.any_horizontal(nwp.exclude("a", "b"), ignore_nulls=False),
            {"c": [True, True, None]},
        ),
        pytest.param(
            nwp.any_horizontal(ncs.all() - ncs.by_index(0, 1), ignore_nulls=True),
            {"c": [True, True, False]},
            id="ignore_nulls-2",
        ),
    ],
)
def test_any_horizontal(
    data: Data, expr: nwp.Expr, expected: Data, dataframe: DataFrame
) -> None:
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)


def test_any_horizontal_kleene_full_null(
    dataframe: DataFrame, request: pytest.FixtureRequest
) -> None:
    data = {"i": [None, None, None]}
    exprs = [
        nwp.any_horizontal(nwp.lit(None, nw.Boolean), "i").alias("None-None"),
        nwp.any_horizontal(nwp.lit(True), "i").alias("True-None"),
        nwp.any_horizontal(nwp.lit(False), "i").alias("False-None"),
    ]
    expected = {
        "None-None": [None, None, None],
        "True-None": [True, True, True],
        "False-None": [None, None, None],
    }
    dataframe.xfail(
        request,
        dataframe.is_pyarrow(),
        reason="`pyarrow` uses `pa.null()`, which also fails in current `narwhals`.\n"
        "In `polars`, the same op is supported and it uses `pl.Null`.\n\n"
        "Function 'or_kleene' has no kernel matching input types (bool, null)",
        raises=NotImplementedError,
    )
    result = dataframe(data).select(exprs)
    assert_equal_data(result, expected)
