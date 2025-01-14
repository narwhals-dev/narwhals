from __future__ import annotations

from typing import Literal

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("left", [True, True, True, False]),
        ("right", [False, True, True, True]),
        ("both", [True, True, True, True]),
        ("none", [False, True, True, False]),
    ],
)
def test_is_between(
    constructor: Constructor,
    closed: Literal["left", "right", "none", "both"],
    expected: list[bool],
) -> None:
    data = {"a": [1, 4, 2, 5]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").is_between(1, 5, closed=closed))
    expected_dict = {"a": expected}
    assert_equal_data(result, expected_dict)


def test_is_between_expressified(constructor: Constructor) -> None:
    data = {"a": [1, 4, 2, 5], "b": [0, 5, 2, 4], "c": [9, 9, 9, 9]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").is_between(nw.col("b") * 0.9, nw.col("c") - 1))
    expected_dict = {"a": [True, False, True, True]}
    assert_equal_data(result, expected_dict)


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("left", [True, True, True, False]),
        ("right", [False, True, True, True]),
        ("both", [True, True, True, True]),
        ("none", [False, True, True, False]),
    ],
)
def test_is_between_series(
    constructor_eager: ConstructorEager,
    closed: Literal["left", "right", "none", "both"],
    expected: list[bool],
) -> None:
    data = {"a": [1, 4, 2, 5]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.with_columns(a=df["a"].is_between(1, 5, closed=closed))
    expected_dict = {"a": expected}
    assert_equal_data(result, expected_dict)


def test_is_between_expressified_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 4, 2, 5], "b": [0, 5, 2, 4], "c": [9, 9, 9, 9]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].is_between(df["b"], df["c"]).to_frame()
    expected_dict = {"a": [True, False, True, True]}
    assert_equal_data(result, expected_dict)
