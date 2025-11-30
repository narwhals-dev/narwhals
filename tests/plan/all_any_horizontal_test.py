from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data


# test_allh_iterator has a different length
@pytest.fixture(scope="module")
def data() -> Data:
    return {
        # test_allh, test_allh_series, test_allh_all, test_allh_nth, test_horizontal_expressions_empty
        "a": [False, False, True],
        "b": [False, True, True],
        # test_all_ignore_nulls, test_allh_kleene, test_anyh_dask,
        "c": [True, True, False],
        "d": [True, None, None],
        "e": [None, True, False],
    }


XFAIL_NOT_IMPL = pytest.mark.xfail(
    reason="TODO: `{all,any}_horizontal(ignore_nulls=True)`", raises=AssertionError
)


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
            marks=XFAIL_NOT_IMPL,
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
            marks=XFAIL_NOT_IMPL,
        ),
    ],
)
def test_all_horizontal(data: Data, expr: nwp.Expr, expected: Data) -> None:
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
            marks=XFAIL_NOT_IMPL,
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
            marks=XFAIL_NOT_IMPL,
        ),
    ],
)
def test_any_horizontal(data: Data, expr: nwp.Expr, expected: Data) -> None:
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)
