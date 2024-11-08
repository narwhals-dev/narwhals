from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_arg_true(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, None, None, 3]}))
    result = df.select(nw.col("a").is_null().arg_true())
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)


def test_arg_true_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, None, None, 3]}), eager_only=True)
    result = df.select(df["a"].is_null().arg_true())
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)
    assert "a" in df  # cheeky test to hit `__contains__` method
