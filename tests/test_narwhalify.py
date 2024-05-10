from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import Any
from typing import ContextManager

import polars as pl
import pytest

import narwhals as nw
from narwhals import DataFrame


@nw.narwhalify
def join_on_key(left: DataFrame, right: DataFrame, key: str) -> DataFrame:
    return left.join(right, left_on=key, right_on=key)


@nw.narwhalify(to_kwargs={"strict": False})
def join_on_key_custom_kwargs(left: DataFrame, right: DataFrame, key: str) -> DataFrame:
    return left.join(right, left_on=key, right_on=key)


frame1 = pl.DataFrame({"a": [1, 1, 2], "b": [0, 1, 2]})
frame2 = pl.DataFrame({"a": [1, 2], "c": ["x", "y"]})
key = "a"


@pytest.mark.parametrize(
    ("args", "kwargs", "context"),
    [
        ((frame1, frame2), {"key": key}, does_not_raise()),
        (
            (frame1, frame2, key),
            {},
            pytest.raises(
                TypeError,
                match="Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe",
            ),
        ),
        (
            (frame1,),
            {"key": key, "right": frame2},
            pytest.raises(
                AttributeError, match="'DataFrame' object has no attribute '_is_polars'"
            ),
        ),
    ],
)
def test_narwhalify(
    args: list[Any], kwargs: dict[str, Any], context: ContextManager[Any]
) -> None:
    with context:
        assert join_on_key(*args, **kwargs) is not None
        assert join_on_key_custom_kwargs(*args, **kwargs) is not None
