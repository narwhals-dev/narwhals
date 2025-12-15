from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Final

import pytest

import narwhals as nw
import narwhals._plan as nwp
from narwhals._plan._dispatch import get_dispatch_name
from narwhals._utils import zip_strict
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from _pytest.mark import ParameterSet
    from typing_extensions import TypeAlias

    from narwhals._plan.typing import OneOrIterable
    from tests.conftest import Data
    from tests.plan.utils import SubList

    SubListNumeric: TypeAlias = SubList[int] | SubList[float]


R1: Final[SubListNumeric] = [3, None, 2, 2, 4, None]
R2: Final[SubListNumeric] = [-1]
R3: Final[SubListNumeric] = None
R4: Final[SubListNumeric] = [None, None, None]
R5: Final[SubListNumeric] = []
# NOTE: `pyarrow` needs at least 3 (non-null) values to calculate `median` correctly
# Otherwise it picks the lowest non-null
# https://github.com/narwhals-dev/narwhals/pull/3332#discussion_r2617508167
R6: Final[SubListNumeric] = [3, 4, None, 4, None, 3]

ROWS: Final[tuple[SubListNumeric, ...]] = R1, R2, R3, R4, R5, R6

EXPECTED_MAX = [4, -1, None, None, None, 4]
EXPECTED_MEAN = [2.75, -1, None, None, None, 3.5]
EXPECTED_MIN = [2, -1, None, None, None, 3]
EXPECTED_SUM = [11, -1, None, 0, 0, 14]
EXPECTED_MEDIAN = [2.5, -1, None, None, None, 3.5]


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [R1, R2, R3, R4, R5, R6]}


a = nwp.col("a")
cast_a = a.cast(nw.List(nw.Int32))


@pytest.mark.parametrize(
    ("exprs", "expected"),
    [
        (a.list.max(), EXPECTED_MAX),
        (a.list.mean(), EXPECTED_MEAN),
        (a.list.min(), EXPECTED_MIN),
        (a.list.sum(), EXPECTED_SUM),
        (a.list.median(), EXPECTED_MEDIAN),
    ],
    ids=["max", "mean", "min", "sum", "median"],
)
def test_list_agg(
    data: Data, exprs: OneOrIterable[nwp.Expr], expected: list[float | None]
) -> None:
    df = dataframe(data).with_columns(cast_a)
    result = df.select(exprs)
    assert_equal_data(result, {"a": expected})


def cases_scalar(
    expr: nwp.Expr, expected: Sequence[float | None]
) -> Iterator[ParameterSet]:
    for idx, row_expected in enumerate(zip_strict(ROWS, expected), start=1):
        row, out = row_expected
        name = get_dispatch_name(expr._ir).removeprefix("list.")
        yield pytest.param(row, expr, out, id=f"{name}-R{idx}")


@pytest.mark.parametrize(
    ("row", "expr", "expected"),
    chain(
        cases_scalar(a.first().list.max(), EXPECTED_MAX),
        cases_scalar(a.first().list.mean(), EXPECTED_MEAN),
        cases_scalar(a.first().list.min(), EXPECTED_MIN),
        cases_scalar(a.first().list.sum(), EXPECTED_SUM),
        cases_scalar(a.first().list.median(), EXPECTED_MEDIAN),
    ),
)
def test_list_agg_scalar(
    row: SubListNumeric, expr: nwp.Expr, expected: float | None
) -> None:
    data = {"a": [row]}
    df = dataframe(data).select(cast_a)
    result = df.select(expr)
    # NOTE: Doing a pure noop on `<pyarrow.ListScalar: None>` will pass `assert_equal_data`,
    # but will have the wrong dtype when compared with a non-null agg
    assert result.collect_schema()["a"] != nw.List
    assert_equal_data(result, {"a": [expected]})
