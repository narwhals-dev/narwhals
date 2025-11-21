from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from narwhals import _plan as nwp
from narwhals._plan import selectors as ncs
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from narwhals._plan.typing import OneOrIterable
    from tests.conftest import Data

DATA_1 = {
    "a": [0.0, None, 2.0, 3.0, 4.0],
    "b": [1.0, None, None, 5.0, 3.0],
    "c": [5.0, None, 3.0, 2.0, 1.0],
}
DATA_2 = {
    "a": [0.0, None, 2.0, 3.0, 4.0],
    "b": [1.0, None, None, 5.0, 3.0],
    "c": [5.0, 2.0, None, 2.0, 1.0],
}
DATA_LIMITS = {
    "a": [1, None, None, None, 5, 6, None, None, None, 10],
    "b": ["a", None, None, None, "b", "c", None, None, None, "d"],
    "idx": list(range(10)),
}


@pytest.mark.parametrize(
    ("data", "exprs", "expected"),
    [
        (  # test_fill_null
            DATA_1,
            nwp.all().fill_null(value=99),
            {"a": [0.0, 99, 2, 3, 4], "b": [1.0, 99, 99, 5, 3], "c": [5.0, 99, 3, 2, 1]},
        ),
        (  # test_fill_null_w_aggregate
            {"a": [0.5, None, 2.0, 3.0, 4.5], "b": ["xx", "yy", "zz", None, "yy"]},
            [nwp.col("a").fill_null(nwp.col("a").mean()), nwp.col("b").fill_null("a")],
            {"a": [0.5, 2.5, 2.0, 3.0, 4.5], "b": ["xx", "yy", "zz", "a", "yy"]},
        ),
        (  # test_fill_null_series_expression
            DATA_2,
            nwp.nth(0, 1).fill_null(nwp.col("c")),
            {"a": [0.0, 2, 2, 3, 4], "b": [1.0, 2, None, 5, 3]},
        ),
        (  # test_fill_null_strategies_with_limit_as_none (1)
            DATA_LIMITS,
            ncs.by_index(0, 1).fill_null(strategy="forward").over(order_by="idx"),
            {
                "a": [1, 1, 1, 1, 5, 6, 6, 6, 6, 10],
                "b": ["a", "a", "a", "a", "b", "c", "c", "c", "c", "d"],
            },
        ),
        (  # test_fill_null_strategies_with_limit_as_none (2)
            DATA_LIMITS,
            nwp.exclude("idx").fill_null(strategy="backward").over(order_by="idx"),
            {
                "a": [1, 5, 5, 5, 5, 6, 10, 10, 10, 10],
                "b": ["a", "b", "b", "b", "b", "c", "d", "d", "d", "d"],
            },
        ),
        (  # test_fill_null_limits (1)
            DATA_LIMITS,
            nwp.col("a", "b").fill_null(strategy="forward", limit=2).over(order_by="idx"),
            {
                "a": [1, 1, 1, None, 5, 6, 6, 6, None, 10],
                "b": ["a", "a", "a", None, "b", "c", "c", "c", None, "d"],
            },
        ),
        (  # test_fill_null_limits (2)
            DATA_LIMITS,
            nwp.col("a", "b")
            .fill_null(strategy="backward", limit=2)
            .over(order_by="idx"),
            {
                "a": [1, None, 5, 5, 5, 6, None, 10, 10, 10],
                "b": ["a", None, "b", "b", "b", "c", None, "d", "d", "d"],
            },
        ),
    ],
)
def test_fill_null(data: Data, exprs: OneOrIterable[nwp.Expr], expected: Data) -> None:
    df = dataframe(data)
    assert_equal_data(df.select(exprs), expected)
