from __future__ import annotations

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


def test_get_column(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2], "b": [3, 4]}), eager_only=True)
    result = df.get_column("a")
    assert_equal_data({"a": result}, {"a": [1, 2]})
    assert result.name == "a"
    with pytest.raises(
        (KeyError, TypeError), match="Expected str|'int' object cannot be converted|0"
    ):
        # Check that trying to get a column by position is disallowed here.
        nw.from_native(df, eager_only=True).get_column(0)  # type: ignore[arg-type]


def test_non_string_name() -> None:
    df = pd.DataFrame({0: [1, 2]})
    result = nw.from_native(df, eager_only=True).get_column(0)  # type: ignore[arg-type]
    assert_equal_data({"a": result}, {"a": [1, 2]})
    assert result.name == 0  # type: ignore[comparison-overlap]


def test_get_single_row() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = nw.from_native(df, eager_only=True)[0]
    assert_equal_data(result, {"a": [1], "b": [3]})
