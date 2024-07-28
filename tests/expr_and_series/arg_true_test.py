from typing import Any

import narwhals as nw
from tests.utils import compare_dicts


def test_arg_true(constructor_lazy: Any) -> None:
    df = nw.from_native(constructor_lazy({"a": [1, None, None, 3]}))
    result = df.select(nw.col("a").is_null().arg_true())
    expected = {"a": [1, 2]}
    compare_dicts(result, expected)


def test_arg_true_series(constructor: Any) -> None:
    df = nw.from_native(constructor({"a": [1, None, None, 3]}), eager_only=True)
    result = df.select(df["a"].is_null().arg_true())
    expected = {"a": [1, 2]}
    compare_dicts(result, expected)
    assert "a" in df  # cheeky test to hit `__contains__` method
