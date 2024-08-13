from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import Any

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError as PlColumnNotFoundError

import narwhals.stable.v1 as nw
from narwhals._exceptions import ColumnNotFoundError
from narwhals.utils import parse_version


@pytest.mark.parametrize(
    ("to_drop", "expected"),
    [
        ("abc", ["b", "z"]),
        (["abc"], ["b", "z"]),
        (["abc", "b"], ["z"]),
        ([nw.selectors.string()], ["abc", "z"]),
        ([nw.selectors.by_dtype(nw.Float64), "abc"], ["b"]),
    ],
)
def test_drop(constructor: Any, to_drop: list[str], expected: list[str]) -> None:
    data = {"abc": [1, 3, 2], "b": ["x", "y", "z"], "z": [7.1, 8, 9]}
    df = nw.from_native(constructor(data))
    assert df.drop(to_drop).collect_schema().names() == expected
    if not isinstance(to_drop, str):
        assert df.drop(*to_drop).collect_schema().names() == expected


@pytest.mark.parametrize(
    ("strict", "context"),
    [
        (
            True,
            pytest.raises(
                (ColumnNotFoundError, PlColumnNotFoundError), match='"z" not found'
            ),
        ),
        (False, does_not_raise()),
    ],
)
def test_drop_strict(request: Any, constructor: Any, strict: bool, context: Any) -> None:  # noqa: FBT001
    if (
        "polars_lazy" in str(request)
        and parse_version(pl.__version__) < (1, 0, 0)
        and strict
    ):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 3, 2], "b": [4, 4, 6]}
    to_drop = ["a", "z"]

    df = nw.from_native(constructor(data))

    with context:
        names_out = df.drop(to_drop, strict=strict).collect_schema().names()
        assert names_out == ["b"]
