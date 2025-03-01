from __future__ import annotations

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_arg_true(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, None, None, 3]}))
    result = df.select(nw_v1.col("a").is_null().arg_true())
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)

    with pytest.deprecated_call(
        match="is deprecated and will be removed in a future version"
    ):
        df.select(nw.col("a").is_null().arg_true())


def test_arg_true_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, None, None, 3]}), eager_only=True)
    result = df.select(df["a"].is_null().arg_true())
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)
    assert "a" in df  # cheeky test to hit `__contains__` method
