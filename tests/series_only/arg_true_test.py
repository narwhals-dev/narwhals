from __future__ import annotations

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


def test_arg_true_series(nw_eager_constructor: ConstructorEager) -> None:
    df = nw.from_native(nw_eager_constructor({"a": [1, None, None, 3]}), eager_only=True)
    result = df.select(df["a"].is_null().arg_true())
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)
    assert "a" in df  # cheeky test to hit `__contains__` method
