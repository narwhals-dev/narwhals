from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._plan import selectors as ncs
from narwhals._utils import zip_strict
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from narwhals._plan.typing import ColumnNameOrSelector
    from tests.conftest import Data


@pytest.fixture
def data() -> Data:
    return {"a": ["a", "b", "a", "b", "c"], "b": [1, 2, 1, 3, 3], "c": [5, 4, 3, 2, 1]}


@pytest.mark.parametrize(
    ("include_key", "expected"),
    [
        (
            True,
            [
                {"a": ["a", "a"], "b": [1, 1], "c": [5, 3]},
                {"a": ["b", "b"], "b": [2, 3], "c": [4, 2]},
                {"a": ["c"], "b": [3], "c": [1]},
            ],
        ),
        (
            False,
            [
                {"b": [1, 1], "c": [5, 3]},
                {"b": [2, 3], "c": [4, 2]},
                {"b": [3], "c": [1]},
            ],
        ),
    ],
    ids=["include_key", "exclude_key"],
)
@pytest.mark.parametrize(
    "by",
    ["a", ncs.string(), ncs.matches("a"), ncs.by_name("a"), ncs.by_dtype(nw.String)],
    ids=["str", "ncs.string", "ncs.matches", "ncs.by_name", "ncs.by_dtype"],
)
def test_partition_by_single(
    data: Data, by: ColumnNameOrSelector, *, include_key: bool, expected: Any
) -> None:
    df = dataframe(data)
    results = df.partition_by(by, include_key=include_key)
    for df, expect in zip_strict(results, expected):
        assert_equal_data(df, expect)
