from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [1.12345, 2.56789, 3.901234, -0.5], "b": [1.045, None, 2.221, -5.9446]}


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwp.col("a").ceil(), [2.0, 3.0, 4.0, 0.0]),
        (nwp.col("a").floor(), [1.0, 2.0, 3.0, -1.0]),
        (nwp.col("b").ceil(), [2.0, None, 3.0, -5.0]),
        (nwp.col("b").floor(), [1.0, None, 2.0, -6.0]),
    ],
    ids=["ceil", "floor", "ceil-nulls", "floor-nulls"],
)
def test_ceil_floor(data: Data, expr: nwp.Expr, expected: list[Any]) -> None:
    result = dataframe(data).select(result=expr)
    assert_equal_data(result, {"result": expected})
