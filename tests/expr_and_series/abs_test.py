from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


def test_abs(nw_frame_constructor: Constructor) -> None:
    df = nw.from_native(nw_frame_constructor({"a": [1, 2, 3, -4, 5]}))
    result = df.select(b=nw.col("a").abs())
    expected = {"b": [1, 2, 3, 4, 5]}
    assert_equal_data(result, expected)


def test_abs_series(nw_eager_constructor: ConstructorEager) -> None:
    df = nw.from_native(nw_eager_constructor({"a": [1, 2, 3, -4, 5]}), eager_only=True)
    result = {"b": df["a"].abs()}
    expected = {"b": [1, 2, 3, 4, 5]}
    assert_equal_data(result, expected)
