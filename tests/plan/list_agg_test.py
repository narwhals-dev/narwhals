from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from narwhals._plan.typing import OneOrIterable
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [[3, None, 2, 2, 4, None], [-1], None, [None, None, None], []]}


@pytest.fixture(scope="module")
def data_median(data: Data) -> Data:
    # NOTE: `pyarrow` needs at least 3 (non-null) values to calculate `median` correctly
    # Otherwise it picks the lowest non-null
    # https://github.com/narwhals-dev/narwhals/pull/3332#discussion_r2617508167
    return {"a": [*data["a"], [3, 4, None, 4, None, 3]]}


a = nwp.col("a")
cast_a = a.cast(nw.List(nw.Int32))


@pytest.mark.parametrize(
    ("exprs", "expected"),
    [
        (a.list.max(), {"a": [4, -1, None, None, None]}),
        (a.list.mean(), {"a": [2.75, -1, None, None, None]}),
        (a.list.min(), {"a": [2, -1, None, None, None]}),
        (a.list.sum(), {"a": [11, -1, None, 0, 0]}),
    ],
    ids=["max", "mean", "min", "sum"],
)
def test_list_agg(data: Data, exprs: OneOrIterable[nwp.Expr], expected: Data) -> None:
    df = dataframe(data).with_columns(cast_a)
    result = df.select(exprs)
    assert_equal_data(result, expected)


def test_list_median(data_median: Data) -> None:
    df = dataframe(data_median).with_columns(cast_a)
    result = df.select(a.list.median())
    assert_equal_data(result, {"a": [2.5, -1, None, None, None, 3.5]})
