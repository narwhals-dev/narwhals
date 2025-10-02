from __future__ import annotations

import pytest

from narwhals import _plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

XFAIL_NO_IMPL_YET = pytest.mark.xfail(
    reason="Not added `ArrowExpr.is_first_distinct` yet", raises=NotImplementedError
)


@XFAIL_NO_IMPL_YET
def test_is_first_distinct_expr() -> None:  # pragma: no cover
    data = {"a": [1, 1, 2, 3, 2], "b": [1, 2, 3, 2, 1]}
    df = dataframe(data)
    result = df.select(nwp.all().is_first_distinct())
    expected = {
        "a": [True, False, True, True, False],
        "b": [True, True, True, False, False],
    }
    assert_equal_data(result, expected)


@XFAIL_NO_IMPL_YET
def test_is_first_distinct_expr_order_by() -> None:  # pragma: no cover
    data = {"a": [1, 1, 2, 3, 2], "b": [1, 2, 3, 2, 1], "i": [None, 1, 2, 3, 4]}
    df = dataframe(data)
    result = (
        df.select(nwp.col("a", "b").is_first_distinct().over(order_by="i"), "i")
        .sort("i")
        .drop("i")
    )
    expected = {
        "a": [True, False, True, True, False],
        "b": [True, True, True, False, False],
    }
    assert_equal_data(result, expected)
