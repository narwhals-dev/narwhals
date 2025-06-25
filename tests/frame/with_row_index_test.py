from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Sequence

data = {"abc": ["foo", "bars"], "xyz": [100, 200], "const": [42, 42]}


def test_with_row_index_eager(constructor_eager: ConstructorEager) -> None:
    result = nw.from_native(constructor_eager(data), eager_only=True).with_row_index()
    expected = {"index": [0, 1], **data}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("order_by", "expected_index"),
    [
        ("abc", [1, 0]),
        ("xyz", [0, 1]),
        (["const", "abc"], [1, 0]),
        (["const", "xyz"], [0, 1]),
    ],
)
def test_with_row_index_lazy(
    constructor: Constructor, order_by: str | Sequence[str], expected_index: list[int]
) -> None:
    if (
        "pandas" in str(constructor) and PANDAS_VERSION < (1, 3) and order_by == "abc"
    ):  # pragma: no cover
        reason = "ValueError: first not supported for non-numeric data."
        pytest.skip(reason=reason)

    result = (
        nw.from_native(constructor(data))
        .with_row_index(name="foo bar", order_by=order_by)
        .sort("xyz")
    )
    expected = {"foo bar": expected_index, **data}
    assert_equal_data(result, expected)


def test_with_row_index_lazy_exception(constructor: Constructor) -> None:
    frame = nw.from_native(constructor(data))

    msg = r"(LazyFrame\.)?with_row_index\(\) missing 1 required keyword-only argument: 'order_by'$"
    context = (
        pytest.raises(TypeError, match=msg)
        if isinstance(frame, nw.LazyFrame)
        else does_not_raise()
    )

    with context:
        result = frame.with_row_index()

        expected = {"index": [0, 1], **data}
        assert_equal_data(result, expected)
