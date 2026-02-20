from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, get_args

import pytest

import narwhals._plan as nwp
from narwhals.typing import ConcatMethod
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from tests.conftest import Data


@pytest.fixture(scope="module")
def left() -> Data:
    return {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


@pytest.fixture(scope="module")
def right() -> Data:
    return {"c": [6, 12, -1], "d": [0, -4, 2]}


def test_concat_horizontal(left: Data, right: Data) -> None:
    result = nwp.concat((dataframe(left), dataframe(right)), how="horizontal")
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "c": [6, 12, -1],
        "d": [0, -4, 2],
    }
    assert_equal_data(result, expected)


def test_concat_vertical(left: Data, right: Data) -> None:
    df_left = dataframe(left).rename({"a": "c", "b": "d"}).drop("z")
    df_right = dataframe(right)
    result = nwp.concat([df_left, df_right], how="vertical")
    expected = {"c": [1, 3, 2, 6, 12, -1], "d": [4, 4, 6, 0, -4, 2]}
    assert_equal_data(result, expected)
    with pytest.raises(
        (Exception, TypeError),
        match=r"unable to vstack|inputs should all have the same schema",
    ):
        nwp.concat([df_left, df_right.rename({"d": "i"})])
    with pytest.raises(
        (Exception, TypeError),
        match=r"unable to vstack|unable to append|inputs should all have the same schema",
    ):
        nwp.concat([df_left, df_left.select("d")], how="vertical")


def test_concat_diagonal() -> None:
    data_1 = {"a": [1, 3], "b": [4, 6]}
    data_2 = {"a": [100, 200], "z": ["x", "y"]}
    expected = {
        "a": [1, 3, 100, 200],
        "b": [4, 6, None, None],
        "z": [None, None, "x", "y"],
    }
    result = nwp.concat([dataframe(data_1), dataframe(data_2)], how="diagonal")
    assert_equal_data(result, expected)


@pytest.mark.parametrize("how", get_args(ConcatMethod))
@pytest.mark.parametrize(
    "into_iterable", [list, tuple, deque, pytest.param(lambda: iter([]), id="iterator")]
)
def test_concat_empty_invalid(
    into_iterable: Callable[[], Iterable[Any]], how: ConcatMethod
) -> None:
    with pytest.raises(ValueError, match="Cannot concatenate an empty iterable"):
        nwp.concat(into_iterable(), how=how)
