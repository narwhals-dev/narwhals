from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.testing.typing import Constructor, ConstructorEager


def test_abs(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3, -4, 5]}))
    result = df.select(b=nw.col("a").abs())
    expected = {"b": [1, 2, 3, 4, 5]}
    assert_equal_data(result, expected)


def test_abs_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3, -4, 5]}), eager_only=True)
    result = {"b": df["a"].abs()}
    expected = {"b": [1, 2, 3, 4, 5]}
    assert_equal_data(result, expected)
