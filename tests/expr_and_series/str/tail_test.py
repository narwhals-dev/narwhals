from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": ["foo", "bars", "東京", "", None]}

tail_cases = [
    (0, {"a": ["", "", "", "", None]}),
    (1, {"a": ["o", "s", "京", "", None]}),
    (3, {"a": ["foo", "ars", "東京", "", None]}),
    (10, data),
]


@pytest.mark.parametrize(("n", "expected"), tail_cases)
def test_str_tail(
    constructor: Constructor, n: int, expected: dict[str, list[str | None]]
) -> None:
    df = nw.from_native(constructor(data))

    result_frame = df.select(nw.col("a").str.tail(n))
    assert_equal_data(result_frame, expected)


@pytest.mark.parametrize(("n", "expected"), tail_cases)
def test_str_tail_series(
    constructor_eager: ConstructorEager, n: int, expected: dict[str, list[str | None]]
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.tail(n)
    assert_equal_data({"a": result_series}, expected)
