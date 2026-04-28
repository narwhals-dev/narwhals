from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._plan import Selector, selectors as ncs
from narwhals._utils import zip_strict
from narwhals.exceptions import ColumnNotFoundError, DuplicateError
from tests.plan.utils import assert_equal_data, dataframe, re_compile

if TYPE_CHECKING:
    from narwhals._plan.typing import ColumnNameOrSelector, OneOrIterable
    from tests.conftest import Data


@pytest.fixture
def data() -> Data:
    return {
        "a": ["a", "b", "a", None, "b", "c"],
        "b": [1, 2, 1, 5, 3, 3],
        "c": [5, 4, 3, 6, 2, 1],
    }


@pytest.mark.parametrize(
    ("include_key", "expected"),
    [
        (
            True,
            [
                {"a": ["a", "a"], "b": [1, 1], "c": [5, 3]},
                {"a": ["b", "b"], "b": [2, 3], "c": [4, 2]},
                {"a": [None], "b": [5], "c": [6]},
                {"a": ["c"], "b": [3], "c": [1]},
            ],
        ),
        (
            False,
            [
                {"b": [1, 1], "c": [5, 3]},
                {"b": [2, 3], "c": [4, 2]},
                {"b": [5], "c": [6]},
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


@pytest.mark.parametrize(
    ("include_key", "expected"),
    [
        (
            True,
            [
                {"a": ["a", "a"], "b": [1, 1], "c": [5, 3]},
                {"a": ["b"], "b": [2], "c": [4]},
                {"a": [None], "b": [5], "c": [6]},
                {"a": ["b"], "b": [3], "c": [2]},
                {"a": ["c"], "b": [3], "c": [1]},
            ],
        ),
        (False, [{"c": [5, 3]}, {"c": [4]}, {"c": [6]}, {"c": [2]}, {"c": [1]}]),
    ],
    ids=["include_key", "exclude_key"],
)
@pytest.mark.parametrize(
    ("by", "more_by"),
    [
        ("a", "b"),
        (["a", "b"], ()),
        (ncs.matches("a|b"), ()),
        (ncs.string(), "b"),
        (ncs.by_name("a", "b"), ()),
        (ncs.by_name("b"), ncs.by_name("a")),
        (ncs.by_dtype(nw.String) | (ncs.numeric() - ncs.by_name("c")), []),
    ],
    ids=[
        "str-variadic",
        "str-list",
        "ncs.matches",
        "ncs.string-str",
        "ncs.by_name",
        "2x-selector",
        "BinarySelector",
    ],
)
def test_partition_by_multiple(
    data: Data,
    by: ColumnNameOrSelector,
    more_by: OneOrIterable[ColumnNameOrSelector],
    *,
    include_key: bool,
    expected: Any,
) -> None:
    df = dataframe(data)
    if isinstance(more_by, (str, Selector)):
        results = df.partition_by(by, more_by, include_key=include_key)
    else:
        results = df.partition_by(by, *more_by, include_key=include_key)
    for df, expect in zip_strict(results, expected):
        assert_equal_data(df, expect)


def test_partition_by_missing_names(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises(ColumnNotFoundError, match=re.escape("not found: ['d']")):
        df.partition_by("d")
    with pytest.raises(ColumnNotFoundError, match=re.escape("not found: ['e']")):
        df.partition_by("c", "e")


def test_partition_by_duplicate_names(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises(DuplicateError, match=re_compile(r"expected.+unique.+got.+'c'")):
        df.partition_by("c", ncs.numeric())


def test_partition_by_fully_empty_selector(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises(
        ColumnNotFoundError, match=re_compile(r"ncs.array.+ncs.struct.+ncs.duration")
    ):
        df.partition_by(ncs.array(ncs.numeric()), ncs.struct(), ncs.duration())


# NOTE: Matching polars behavior
def test_partition_by_partially_missing_selector(data: Data) -> None:
    df = dataframe(data)
    results = df.partition_by(ncs.string() | ncs.list() | ncs.enum())
    expected = nw.Schema({"a": nw.String(), "b": nw.Int64(), "c": nw.Int64()})
    for df in results:
        assert df.schema == expected
