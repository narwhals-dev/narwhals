from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": ["Starts_with", "starts_with", "Ends_with", "ends_With", None]}


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("start", [False, True, False, False, None]),
        ("End", [False, False, True, False, None]),
    ],
)
def test_str_starts_with(data: Data, prefix: str, expected: list[bool | None]) -> None:
    result = dataframe(data).select(nwp.col("a").str.starts_with(prefix))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [("With", [False, False, False, True, None]), ("th", [True, True, True, True, None])],
)
def test_str_ends_with(data: Data, suffix: str, expected: list[bool | None]) -> None:
    result = dataframe(data).select(nwp.col("a").str.ends_with(suffix))
    assert_equal_data(result, {"a": expected})
