from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from narwhals._plan.typing import OneOrIterable
    from tests.conftest import Data

pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [[1, 2], [3, 4, None], None, [], [None]], "i": [4, 3, 2, 1, 0]}


a = nwp.nth(0)


@pytest.mark.parametrize(
    ("exprs", "expected"),
    [
        (a.list.len(), {"a": [2, 3, None, 0, 1]}),
        (
            [a.first().list.len().alias("first"), a.last().list.len().alias("last")],
            {"first": [2], "last": [1]},
        ),
        (  # NOTE: `polars` produces nulls following the `over(order_by=...)`
            # That's either a bug, or something that won't be ported to `narwhals`
            [
                a.first().over(order_by="i").list.len().alias("first_order_i"),
                a.last().over(order_by="i").list.len().alias("last_order_i"),
            ],
            {"first_order_i": [1], "last_order_i": [2]},
        ),
        (
            # NOTE: This does work already in `polars`
            [
                a.sort_by("i").first().list.len().alias("sort_by_i_first"),
                a.sort_by("i").last().list.len().alias("sort_by_i_last"),
            ],
            {"sort_by_i_first": [1], "sort_by_i_last": [2]},
        ),
    ],
)
def test_list_len(data: Data, exprs: OneOrIterable[nwp.Expr], expected: Data) -> None:
    df = dataframe(data).with_columns(a.cast(nw.List(nw.Int32())))
    result = df.select(exprs)
    assert_equal_data(result, expected)
