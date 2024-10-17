from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts


def test_any_all(constructor: Constructor) -> None:
    df = nw.from_native(
        constructor(
            {
                "a": [True, False, True],
                "b": [True, True, True],
                "c": [False, False, False],
            }
        )
    )
    result = df.select(nw.col("a", "b", "c").all())
    expected = {"a": [False], "b": [True], "c": [False]}
    compare_dicts(result, expected)
    result = df.select(nw.all().any())
    expected = {"a": [True], "b": [True], "c": [False]}
    compare_dicts(result, expected)


def test_any_all_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager(
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
    compare_dicts(result, expected)
    result = {"a": [df["a"].any()], "b": [df["b"].any()], "c": [df["c"].any()]}
    expected = {"a": [True], "b": [True], "c": [False]}
    compare_dicts(result, expected)
