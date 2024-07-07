from typing import Any

import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts


def test_lazy_instantiation() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    result = nw.from_native(pl.LazyFrame(data))
    expected = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    compare_dicts(result, expected)


def test_lazy_instantiation_error() -> None:
    df_lazy = pl.LazyFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
    with pytest.raises(
        TypeError, match="Can't instantiate DataFrame from Polars LazyFrame."
    ):
        _ = nw.DataFrame(df_lazy, is_polars=True, backend_version=(0,)).shape


def test_eager_instantiation(constructor_with_pyarrow: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    result = nw.from_native(constructor_with_pyarrow(data), eager_only=True)
    expected = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    compare_dicts(result, expected)
