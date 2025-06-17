from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data

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
@pytest.mark.skipif(
    POLARS_VERSION < (1, 9, 0),
    reason="Too old for `.over(partition_by=...)` or does not break ties with multiple columns in partition_by",
)
def test_with_row_index_lazy(
    constructor: Constructor, order_by: str | Sequence[str], expected_index: list[int]
) -> None:
    result = (
        nw.from_native(constructor(data)).with_row_index(order_by=order_by).sort("xyz")
    )
    expected = {"index": expected_index, **data}
    assert_equal_data(result, expected)


def test_with_row_index_lazy_exception(constructor: Constructor) -> None:
    msg = (
        "`LazyFrame.with_row_index` requires `order_by` to be specified as it is an "
        "order-dependent operation."
    )

    context = (
        pytest.raises(ValueError, match=msg)
        if any(x in str(constructor) for x in ("duckdb", "ibis", "pyspark"))
        else does_not_raise()
    )
    frame = nw.from_native(constructor(data))

    with context:
        frame.with_row_index()
