from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"pets": ["cat", "dog", "rabbit and parrot", "dove", "Parrot|dove", None]}


@pytest.mark.parametrize(
    ("pattern", "literal", "expected"),
    [
        ("(?i)parrot|Dove", False, [False, False, True, True, True, None]),
        ("parrot|Dove", False, [False, False, True, False, False, None]),
        ("Parrot|dove", False, [False, False, False, True, True, None]),
        ("Parrot|dove", True, [False, False, False, False, True, None]),
    ],
    ids=["case_insensitive", "case_sensitive-1", "case_sensitive-2", "literal"],
)
def test_str_contains(
    data: Data, pattern: str, *, literal: bool, expected: list[bool | None]
) -> None:
    result = dataframe(data).select(
        nwp.col("pets").str.contains(pattern, literal=literal)
    )
    assert_equal_data(result, {"pets": expected})
