from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import Any

import polars as pl
import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "ix": [1, 1, 2, 2, 1, 2],
    "col": ["a", "a", "a", "a", "b", "b"],
    "foo": [0, 1, 2, 2, 7, 1],
    "bar": [0, 2, 0, 0, 9, 4],
}

data_no_dups = {
    "ix": [1, 1, 2, 2],
    "col": ["a", "b", "a", "b"],
    "foo": [1, 2, 3, 4],
    "bar": ["x", "y", "z", "w"],
}


@pytest.mark.parametrize(
    ("agg_func", "expected"),
    [
        (
            "min",
            {
                "ix": [1, 2],
                "foo_a": [0, 2],
                "foo_b": [7, 1],
                "bar_a": [0, 0],
                "bar_b": [9, 4],
            },
        ),
        # ("max",),  # noqa: ERA001
        # ("first",),  # noqa: ERA001
        # ("last",),  # noqa: ERA001
        # ("sum",),  # noqa: ERA001
        # ("mean",),  # noqa: ERA001
        # ("median",),  # noqa: ERA001
        # ("len",),  # noqa: ERA001
    ],
)
def test_pivot(
    request: Any, constructor: Any, agg_func: str | None, expected: dict[str, list[Any]]
) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data), eager_only=True)
    result = df.pivot(  # noqa: PD010
        "col",
        index="ix",
        aggregate_function=agg_func,  #   type: ignore[arg-type]
    )

    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("data_", "context"),
    [
        (data_no_dups, does_not_raise()),
        (data, pytest.raises((ValueError, pl.exceptions.ComputeError))),
    ],
)
def test_pivot_none(request: Any, constructor: Any, data_: Any, context: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data_), eager_only=True)
    with context:
        df.pivot("col", index="ix", aggregate_function=None)  # noqa: PD010
