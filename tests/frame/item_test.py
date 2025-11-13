from __future__ import annotations

import re
from typing import Any

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


@pytest.mark.parametrize(("row", "column", "expected"), [(0, 2, 7), (1, "z", 8)])
def test_item(
    constructor_eager: ConstructorEager,
    row: int | None,
    column: int | str | None,
    expected: Any,
) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    assert_equal_data({"a": [df.item(row, column)]}, {"a": [expected]})
    assert_equal_data({"a": [df.select("a").head(1).item()]}, {"a": [1]})


@pytest.mark.parametrize(
    ("row", "column", "err_msg"),
    [
        (0, None, re.escape("cannot call `.item()` with only one of `row` or `column`")),
        (None, 0, re.escape("cannot call `.item()` with only one of `row` or `column`")),
        (
            None,
            None,
            re.escape("can only call `.item()` if the dataframe is of shape (1, 1)"),
        ),
    ],
)
def test_item_value_error(
    constructor_eager: ConstructorEager,
    row: int | None,
    column: int | str | None,
    err_msg: str,
) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    with pytest.raises(ValueError, match=err_msg):
        nw.from_native(constructor_eager(data), eager_only=True).item(row, column)
