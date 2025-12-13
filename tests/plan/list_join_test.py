from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "a": [
            ["a", "b", "c"],
            [None, None, None],
            [None, None, "1", "2", None, "3", None],
            ["x", "y"],
            ["1", None, "3"],
            [None],
            None,
            [],
            [None, None],
        ]
    }


a = nwp.col("a")

# TODO @dangotbanned: Ensure the final branch works when replacements are mixed
XFAIL_INCORRECT_RESULTS = pytest.mark.xfail(
    reason="Returned out-of-order post-join", raises=AssertionError
)


@pytest.mark.parametrize(
    ("separator", "ignore_nulls", "expected"),
    [
        ("-", False, ["a-b-c", None, None, "x-y", None, None, None, "", None]),
        pytest.param(
            "-",
            True,
            ["a-b-c", "", "1-2-3", "x-y", "1-3", "", None, "", ""],
            marks=XFAIL_INCORRECT_RESULTS,
        ),
        ("", False, ["abc", None, None, "xy", None, None, None, "", None]),
        pytest.param(
            "",
            True,
            ["abc", "", "123", "xy", "13", "", None, "", ""],
            marks=XFAIL_INCORRECT_RESULTS,
        ),
    ],
    ids=[
        "hyphen-propagate-nulls",
        "hyphen-ignore-nulls",
        "empty-propagate-nulls",
        "empty-ignore-nulls",
    ],
)
def test_list_join(
    data: Data, separator: str, *, ignore_nulls: bool, expected: list[str | None]
) -> None:
    df = dataframe(data).with_columns(a.cast(nw.List(nw.String)))
    expr = a.list.join(separator, ignore_nulls=ignore_nulls)
    result = df.select(expr)
    assert_equal_data(result, {"a": expected})


@pytest.mark.xfail
def test_list_join_scalar() -> None:  # pragma: no cover
    msg = "TODO: Add non-duplicated tests for this"
    raise NotImplementedError(msg)
