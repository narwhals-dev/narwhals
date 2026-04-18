from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


def test_any_all(nw_frame_constructor: Constructor) -> None:
    df = nw.from_native(
        nw_frame_constructor(
            {
                "a": [True, False, True],
                "b": [True, True, True],
                "c": [False, False, False],
            }
        )
    )
    result = df.select(nw.col("a", "b", "c").all())
    expected = {"a": [False], "b": [True], "c": [False]}
    assert_equal_data(result, expected)
    result = df.select(nw.col("a", "b", "c").any())
    expected = {"a": [True], "b": [True], "c": [False]}
    assert_equal_data(result, expected)


def test_any_all_series(nw_eager_constructor: ConstructorEager) -> None:
    df = nw.from_native(
        nw_eager_constructor(
            {
                "a": [True, False, True],
                "b": [True, True, True],
                "c": [False, False, False],
            }
        ),
        eager_only=True,
    )
    result = {"a": [df["a"].all()], "b": [df["b"].all()], "c": [df["c"].all()]}
    expected = {"a": [False], "b": [True], "c": [False]}
    assert_equal_data(result, expected)
    result = {"a": [df["a"].any()], "b": [df["b"].any()], "c": [df["c"].any()]}
    expected = {"a": [True], "b": [True], "c": [False]}
    assert_equal_data(result, expected)
