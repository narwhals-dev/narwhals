from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import pytest

import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from narwhals.exceptions import ShapeError
from tests.plan.utils import assert_equal_data, assert_equal_series, dataframe, series

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "idx": list(range(10)),
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    }


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("offset", [0, 1, 2, 3])
@pytest.mark.parametrize("column", ["idx", "name"])
def test_gather_every_series(data: Data, n: int, offset: int, column: str) -> None:
    ser = series(data[column]).alias(column)
    result = ser.gather_every(n, offset)
    expected = data[column][offset::n]
    assert_equal_series(result, expected, column)


@pytest.mark.parametrize(
    ("column", "indices", "expected"),
    [
        ("idx", [], []),
        ("name", [], []),
        ("idx", [0, 4, 2], [0, 4, 2]),
        ("name", [1, 5, 5], ["b", "f", "f"]),
        pytest.param(
            "idx",
            [-1],
            [9],
            marks=pytest.mark.xfail(
                reason="TODO: Handle negative indices", raises=IndexError
            ),
        ),
        ("name", range(5, 7), ["f", "g"]),
    ],
)
def test_gather_series(
    data: Data, column: str, indices: Any, expected: list[Any]
) -> None:
    ser = series(data[column]).alias(column)
    result = ser.gather(indices)
    assert_equal_series(result, expected, column)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_gather_every_dataframe(data: Data, n: int, offset: int) -> None:
    result = dataframe(data).gather_every(n, offset)
    indices = slice(offset, None, n)
    expected = {"idx": data["idx"][indices], "name": data["name"][indices]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_gather_every_expr(data: Data, n: int, offset: int) -> None:
    df = dataframe(data)
    indices = slice(offset, None, n)
    v_idx, v_name = data["idx"][indices], data["name"][indices]
    e_idx, e_name = nwp.col("idx"), nwp.col("name")
    gather = partial(nwp.Expr.gather_every, n=n, offset=offset)

    result = df.select(gather(nwp.col("idx", "name")))
    expected = {"idx": v_idx, "name": v_name}
    assert_equal_data(result, expected)
    expected = {"name": v_name}
    assert_equal_data(df.select(gather(e_name)), expected)
    expected = {"name": v_name, "idx": v_idx}
    assert_equal_data(df.select(gather(nwp.nth(1, 0))), expected)
    expected = {"idx": v_idx, "name": v_name}
    assert_equal_data(df.select(gather(e_idx), gather(ncs.last())), expected)

    if n == 1 and offset == 0:
        result = df.select(gather(e_name), e_idx)
        expected = {"name": data["name"], "idx": data["idx"]}
        assert_equal_data(result, expected)
    else:
        with pytest.raises(ShapeError):
            df.select(gather(e_name), e_idx)
        result = df.select(gather(e_name), e_idx.first())
        expected = {"name": v_name, "idx": [0] * len(result)}
        assert_equal_data(result, expected)
