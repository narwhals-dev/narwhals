from __future__ import annotations

import re
from typing import Any

import pytest
from tests.utils import ConstructorEager, assert_equal_data

import narwhals as nw


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
        (0, None, "cannot call `.item()` with only one of `row` or `column`"),
        (None, 0, "cannot call `.item()` with only one of `row` or `column`"),
        (
            None,
            None,
            'can only call `.item()` without "row" or "column" values if the DataFrame has a single element; shape=(3, 3)',
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
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        nw.from_native(constructor_eager(data), eager_only=True).item(row, column)
