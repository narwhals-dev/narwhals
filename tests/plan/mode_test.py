from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from narwhals import _plan as nwp
from narwhals._plan import selectors as ncs
from narwhals.exceptions import ShapeError
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [1, 1, 2, 2, 3], "b": [1, 2, 3, 3, 4]}


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwp.col("b").mode(), {"b": [3]}),
        (nwp.col("a").mode(keep="all"), {"a": [1, 2]}),
        (nwp.col("b").filter(nwp.col("b") != 3).mode(), {"b": [1, 2, 4]}),
        (nwp.col("a").mode().sum(), {"a": [3]}),
    ],
    ids=["single", "multiple-1", "multiple-2", "mutliple-agg"],
)
def test_mode_expr_keep_all(data: Data, expr: nwp.Expr, expected: Data) -> None:
    result = dataframe(data).select(expr).sort(ncs.first())
    assert_equal_data(result, expected)


def test_mode_expr_different_lengths_keep_all(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises(ShapeError):
        df.select(nwp.col("a", "b").mode(keep="all"))


def test_mode_expr_keep_any(data: Data) -> None:
    result = dataframe(data).select(nwp.col("a", "b").mode(keep="any"))
    try:
        expected = {"a": [1], "b": [3]}
        assert_equal_data(result, expected)
    except AssertionError:  # pragma: no cover
        expected = {"a": [2], "b": [3]}
        assert_equal_data(result, expected)
