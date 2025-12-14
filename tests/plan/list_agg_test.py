from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe
from tests.utils import is_windows

if TYPE_CHECKING:
    from narwhals._plan.typing import OneOrIterable
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [[3, None, 2, 2, 4, None], [-1], None, [None, None, None], []]}


@pytest.fixture(scope="module")
def data_median(data: Data) -> Data:
    return {"a": [*data["a"], [3, 4, None]]}


a = nwp.col("a")
cast_a = a.cast(nw.List(nw.Int32))


XFAIL_NOT_IMPL = pytest.mark.xfail(
    reason="TODO: ArrowExpr.list.<agg>", raises=NotImplementedError
)


@XFAIL_NOT_IMPL
@pytest.mark.parametrize(
    ("exprs", "expected"),
    [
        (a.list.max(), {"a": [4, -1, None, None, None]}),
        (a.list.mean(), {"a": [2.75, -1, None, None, None]}),
        (a.list.min(), {"a": [2, -1, None, None, None]}),
        (a.list.sum(), {"a": [11, -1, None, 0, 0]}),
    ],
)
def test_list_agg(
    data: Data, exprs: OneOrIterable[nwp.Expr], expected: Data
) -> None:  # pragma: no cover
    df = dataframe(data).with_columns(cast_a)
    result = df.select(exprs)
    assert_equal_data(result, expected)


@XFAIL_NOT_IMPL
@pytest.mark.xfail(
    is_windows() and sys.version_info < (3, 10), reason="Old pyarrow windows bad?"
)
def test_list_median(data_median: Data) -> None:  # pragma: no cover
    df = dataframe(data_median).with_columns(cast_a)
    result = df.select(a.list.median())

    # TODO @dangotbanned: Is this fixable with `FunctionOptions`?
    expected = [2.5, -1, None, None, None, 3.5]
    expected_pyarrow = [2.5, -1, None, None, None, 3]
    expected = expected_pyarrow
    assert_equal_data(result, {"a": expected})
