from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import Any
from typing import ContextManager

import polars as pl
import pytest

import narwhals as nw


@nw.narwhalify(
    from_kwargs={"eager_only": True},
    to_kwargs={"strict": False},
)
def shape_greater_than(df_any: Any, n: int = 0) -> Any:
    return df_any.shape[0] > n


frame = pl.DataFrame({"a": [1, 1, 2], "b": [0, 1, 2]})


@pytest.mark.parametrize(
    ("args", "kwargs", "context"),
    [
        ((frame,), {}, does_not_raise()),
        ((frame, 5), {}, does_not_raise()),
        ((frame,), {"n": 5}, does_not_raise()),
        (
            (),
            {"df_any": frame, "n": 5},
            pytest.raises(
                TypeError, match="missing 1 required positional argument: 'frame'"
            ),
        ),
        (
            (pl.LazyFrame(frame),),
            {},
            pytest.raises(
                TypeError, match="Cannot only use `eager_only` with polars.LazyFrame"
            ),
        ),
    ],
)
def test_narwhalify(
    args: list[Any], kwargs: dict[str, Any], context: ContextManager[Any]
) -> None:
    with context:
        assert shape_greater_than(*args, **kwargs) is not None
