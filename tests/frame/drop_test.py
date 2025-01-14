from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING
from typing import Any

import pytest
from polars.exceptions import ColumnNotFoundError as PlColumnNotFoundError

import narwhals.stable.v1 as nw
from narwhals.exceptions import ColumnNotFoundError
from tests.utils import POLARS_VERSION

if TYPE_CHECKING:
    from tests.utils import Constructor


@pytest.mark.parametrize(
    ("to_drop", "expected"),
    [
        ("abc", ["b", "z"]),
        (["abc"], ["b", "z"]),
        (["abc", "b"], ["z"]),
    ],
)
def test_drop(constructor: Constructor, to_drop: list[str], expected: list[str]) -> None:
    data = {"abc": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    assert df.drop(to_drop).collect_schema().names() == expected
    if not isinstance(to_drop, str):
        assert df.drop(*to_drop).collect_schema().names() == expected


@pytest.mark.parametrize(
    ("strict", "context"),
    [
        (
            True,
            pytest.raises((ColumnNotFoundError, PlColumnNotFoundError), match="z"),
        ),
        (False, does_not_raise()),
    ],
)
def test_drop_strict(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    context: Any,
    *,
    strict: bool,
) -> None:
    if "polars_lazy" in str(constructor) and POLARS_VERSION < (1, 0, 0) and strict:
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 3, 2], "b": [4, 4, 6]}
    to_drop = ["a", "z"]

    df = nw.from_native(constructor(data))

    with context:
        names_out = df.drop(to_drop, strict=strict).collect_schema().names()
        assert names_out == ["b"]
