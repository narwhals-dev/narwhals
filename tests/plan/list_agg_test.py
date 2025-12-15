from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from narwhals._plan.typing import OneOrIterable
    from tests.conftest import Data
    from tests.plan.utils import SubList


R1: Final[SubList[int]] = [3, None, 2, 2, 4, None]
R2: Final[SubList[int]] = [-1]
R3: Final[SubList[float]] = None
R4: Final[SubList[float]] = [None, None, None]
R5: Final[SubList[float]] = []
# NOTE: `pyarrow` needs at least 3 (non-null) values to calculate `median` correctly
# Otherwise it picks the lowest non-null
# https://github.com/narwhals-dev/narwhals/pull/3332#discussion_r2617508167
R6: Final[SubList[int]] = [3, 4, None, 4, None, 3]


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [R1, R2, R3, R4, R5, R6]}


a = nwp.col("a")
cast_a = a.cast(nw.List(nw.Int32))


@pytest.mark.parametrize(
    ("exprs", "expected"),
    [
        (a.list.max(), [4, -1, None, None, None, 4]),
        (a.list.mean(), [2.75, -1, None, None, None, 3.5]),
        (a.list.min(), [2, -1, None, None, None, 3]),
        (a.list.sum(), [11, -1, None, 0, 0, 14]),
        (a.list.median(), [2.5, -1, None, None, None, 3.5]),
    ],
    ids=["max", "mean", "min", "sum", "median"],
)
def test_list_agg(
    data: Data, exprs: OneOrIterable[nwp.Expr], expected: list[float | None]
) -> None:
    df = dataframe(data).with_columns(cast_a)
    result = df.select(exprs)
    assert_equal_data(result, {"a": expected})


first = a.first()
first_list_max = first.list.max()
first_list_mean = first.list.mean()
first_list_min = first.list.min()
first_list_sum = first.list.sum()
first_list_median = first.list.median()


# TODO @dangotbanned: Shrink this
@pytest.mark.parametrize(
    ("row", "expr", "expected"),
    [
        (R1, first_list_max, 4),
        (R2, first_list_max, -1),
        (R3, first_list_max, None),
        (R4, first_list_max, None),
        (R5, first_list_max, None),
        (R6, first_list_max, 4),
        (R1, first_list_mean, 2.75),
        (R2, first_list_mean, -1),
        (R3, first_list_mean, None),
        (R4, first_list_mean, None),
        (R5, first_list_mean, None),
        (R6, first_list_mean, 3.5),
        (R1, first_list_min, 2),
        (R2, first_list_min, -1),
        (R3, first_list_min, None),
        (R4, first_list_min, None),
        (R5, first_list_min, None),
        (R6, first_list_min, 3),
        (R1, first_list_sum, 11),
        (R2, first_list_sum, -1),
        (R3, first_list_sum, None),
        (R4, first_list_sum, 0),
        (R5, first_list_sum, 0),
        (R6, first_list_sum, 14),
        (R1, first_list_median, 2.5),
        (R2, first_list_median, -1),
        (R3, first_list_median, None),
        (R4, first_list_median, None),
        (R5, first_list_median, None),
        (R6, first_list_median, 3.5),
    ],
)
def test_list_agg_scalar(
    row: SubList[float], expr: nwp.Expr, expected: float | None
) -> None:
    data = {"a": [row]}
    df = dataframe(data).select(cast_a)
    result = df.select(expr)
    # NOTE: Doing a pure noop on `<pyarrow.ListScalar: None>` will pass `assert_equal_data`,
    # but will have the wrong dtype when compared with a non-null agg
    assert result.collect_schema()["a"] != nw.List
    assert_equal_data(result, {"a": [expected]})
