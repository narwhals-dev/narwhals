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
    return {
        "a": [[1, 2], [3, 4, None], None, [None]],
        "b": [[None, "o"], ["b", None, "b"], [None, "oops", None, "hi"], None],
    }


a = nwp.nth(0)
b = nwp.col("b")


@pytest.mark.xfail(reason="TODO: `ArrowExpr.list.get`", raises=NotImplementedError)
@pytest.mark.parametrize(
    ("exprs", "expected"),
    [
        (a.list.get(0), {"a": [1, 3, None, None]}),
        (b.list.get(1), {"b": ["o", None, "oops", None]}),
    ],
)
def test_list_get(
    data: Data, exprs: OneOrIterable[nwp.Expr], expected: Data
) -> None:  # pragma: no cover
    df = dataframe(data).with_columns(
        a.cast(nw.List(nw.Int32())), b.cast(nw.List(nw.String))
    )
    result = df.select(exprs)
    assert_equal_data(result, expected)
