from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

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
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()

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
    if isinstance(frame, nw.LazyFrame):
        with pytest.raises(TypeError, match=msg):
            frame.with_row_index()  # type: ignore[call-arg]
    else:
        result = frame.with_row_index()
        assert_equal_data(result, {"index": [0, 1], **data})


@pytest.mark.parametrize(
    ("order_by", "expected_index"),
    [
        (["a"], [0, 2, 1]),
        (["c"], [2, 0, 1]),
        (["a", "c"], [1, 2, 0]),
        (["c", "a"], [2, 0, 1]),
    ],
)
def test_with_row_index_lazy_meaner_examples(
    constructor: Constructor, order_by: list[str], expected_index: list[int]
) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3289
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    data = {"a": ["A", "B", "A"], "b": [1, 2, 3], "c": [9, 2, 4]}
    df = nw.from_native(constructor(data))
    result = df.with_row_index(name="index", order_by=order_by).sort("b")
    expected = {"index": expected_index, **data}
    assert_equal_data(result, expected)
