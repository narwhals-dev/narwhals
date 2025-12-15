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
    from collections.abc import Iterable, Iterator, Sequence

    from _pytest.mark import ParameterSet

    from narwhals.typing import NonNestedLiteral
    from tests.conftest import Data
    from tests.plan.utils import SubList


ROWS_N: Final[tuple[SubList[int] | SubList[float], ...]] = (
    [3, None, 2, 2, 4, None],
    [-1],
    None,
    [None, None, None],
    [],
    [3, 4, None, 4, None, 3],
)
# NOTE: `pyarrow` needs at least 3 (non-null) values to calculate `median` correctly
# Otherwise it picks the lowest non-null
# https://github.com/narwhals-dev/narwhals/pull/3332#discussion_r2617508167


ROWS_B: Final[tuple[SubList[bool], ...]] = (
    [True, True],
    [False, True],
    [False, False],
    [None],
    [],
    None,
)

EXPECTED_MAX = [4, -1, None, None, None, 4]
EXPECTED_MEAN = [2.75, -1, None, None, None, 3.5]
EXPECTED_MIN = [2, -1, None, None, None, 3]
EXPECTED_SUM = [11, -1, None, 0, 0, 14]
EXPECTED_MEDIAN = [2.5, -1, None, None, None, 3.5]
EXPECTED_ALL = [True, False, False, True, True, None]
EXPECTED_ANY = [True, True, False, False, False, None]


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [*ROWS_N], "b": [*ROWS_B]}


a = nwp.col("a")
b = nwp.col("b")
cast_a = a.cast(nw.List(nw.Int32))
cast_b = b.cast(nw.List(nw.Boolean))


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (a.list.max(), EXPECTED_MAX),
        (a.list.mean(), EXPECTED_MEAN),
        (a.list.min(), EXPECTED_MIN),
        (a.list.sum(), EXPECTED_SUM),
        (a.list.median(), EXPECTED_MEDIAN),
        (b.list.all(), EXPECTED_ALL),
        (b.list.any(), EXPECTED_ANY),
    ],
    ids=["max", "mean", "min", "sum", "median", "all", "any"],
)
def test_list_agg(data: Data, expr: nwp.Expr, expected: list[NonNestedLiteral]) -> None:
    df = dataframe(data).with_columns(cast_a, cast_b)
    result = df.select(result=expr)
    assert_equal_data(result, {"result": expected})


def cases_scalar(
    expr: nwp.Expr,
    rows: Iterable[Sequence[NonNestedLiteral] | None],
    expected: Sequence[NonNestedLiteral],
) -> Iterator[ParameterSet]:
    for idx, row_expected in enumerate(zip_strict(rows, expected), start=1):
        row, out = row_expected
        name = get_dispatch_name(expr._ir).removeprefix("list.")
        yield pytest.param(expr, row, out, id=f"{name}-R{idx}")


first_n = nwp.nth(0).cast(nw.List(nw.Int32)).first()
first_b = nwp.nth(0).cast(nw.List(nw.Boolean)).first()


@pytest.mark.parametrize(
    ("expr", "row", "expected"),
    chain(
        cases_scalar(first_n.list.max(), ROWS_N, EXPECTED_MAX),
        cases_scalar(first_n.list.mean(), ROWS_N, EXPECTED_MEAN),
        cases_scalar(first_n.list.min(), ROWS_N, EXPECTED_MIN),
        cases_scalar(first_n.list.sum(), ROWS_N, EXPECTED_SUM),
        cases_scalar(first_n.list.median(), ROWS_N, EXPECTED_MEDIAN),
        cases_scalar(first_b.list.all(), ROWS_B, EXPECTED_ALL),
        cases_scalar(first_b.list.any(), ROWS_B, EXPECTED_ANY),
    ),
)
def test_list_agg_scalar(
    expr: nwp.Expr, row: SubList[NonNestedLiteral], expected: NonNestedLiteral
) -> None:
    data = {"a": [row]}
    result = dataframe(data).select(expr)
    # NOTE: Doing a pure noop on `<pyarrow.ListScalar: None>` will pass `assert_equal_data`,
    # but will have the wrong dtype when compared with a non-null agg
    assert result.collect_schema()["a"] != nw.List
    assert_equal_data(result, {"a": [expected]})
