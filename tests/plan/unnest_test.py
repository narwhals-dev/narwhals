from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pytest

import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from narwhals.exceptions import ColumnNotFoundError, DuplicateError, InvalidOperationError
from tests.plan.utils import DataFrame, assert_equal_data, re_compile

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._plan.typing import ColumnNameOrSelector, OneOrIterable
    from tests.conftest import Data

BEFORE = "before"
A = "t_a"
B = "t_b"
C = "t_c"
D = "t_d"
AFTER = "after"


@pytest.fixture
def data() -> Data:
    """Dataset from [upstream].

    [upstream]: https://github.com/pola-rs/polars/blob/346a793589efd552a6c10c857e0f0434f7e9a7d4/py-polars/src/polars/lazyframe/frame.py#L9010-L9054
    """
    return {
        BEFORE: ["foo", "bar"],
        A: [1, 2],
        B: ["a", "b"],
        C: [True, None],
        D: [[1, 2], [3]],
        AFTER: ["baz", "womp"],
    }


S = "struct"


@pytest.mark.parametrize(
    ("struct_columns", "select_exprs", "expected_columns"),
    [
        ([C], [S, nwp.exclude(S)], [C, BEFORE, A, B, D, AFTER]),
        ([A, B], [nwp.exclude(S), S], [BEFORE, C, D, AFTER, A, B]),
        ([D, A, B, C], [ncs.first(), S, AFTER], [BEFORE, D, A, B, C, AFTER]),
    ],
    ids=["1-column", "2-column", "4-column"],
)
def test_unnest_frame_single_struct(
    data: Data,
    struct_columns: list[str],
    select_exprs: Sequence[str | nwp.Expr],
    expected_columns: list[str],
    dataframe: DataFrame,
) -> None:

    expected = dataframe(data).select(expected_columns).to_dict(as_series=False)
    df = (
        dataframe(data)
        .with_columns(nwp.struct(struct_columns).alias(S))
        .drop(struct_columns)
        .select(select_exprs)
    )

    assert_equal_data(df.unnest(S), expected, check_column_order=True)
    assert_equal_data(df.unnest(ncs.struct()), expected, check_column_order=True)
    assert_equal_data(df.unnest(~~ncs.matches(S)), expected, check_column_order=True)


def test_unnest_frame_multi_struct(data: Data, dataframe: DataFrame) -> None:
    expected_identical_input = copy.deepcopy(data)
    expected_reorder = {
        BEFORE: ["foo", "bar"],
        C: [True, None],
        D: [[1, 2], [3]],
        A: [1, 2],
        B: ["a", "b"],
        AFTER: ["baz", "womp"],
    }

    name_1 = "t_struct_1"
    name_2 = "t_struct_2"

    df = dataframe(data).select(
        BEFORE, nwp.struct(A, B).alias(name_1), nwp.struct([C, D]).alias(name_2), AFTER
    )
    assert_equal_data(df.unnest(name_1, name_2), expected_identical_input)
    assert_equal_data(df.unnest(ncs.struct()), expected_identical_input)

    # Unnest does not reorder columns
    flipped = ncs.by_index(2, 1)
    assert_equal_data(df.unnest(flipped), expected_identical_input)
    # You'd need to do that yourself
    assert_equal_data(df.select(BEFORE, flipped, AFTER).unnest(flipped), expected_reorder)


def test_unnest_frame_invalid_operation_error(data: Data, dataframe: DataFrame) -> None:
    df = dataframe(data)
    with pytest.raises(
        InvalidOperationError,
        match=re_compile(r"unnest.+not supported for.+string.+expected.+struct"),
    ):
        df.unnest("before")

    with pytest.raises(InvalidOperationError):
        df.unnest(ncs.last().first())


@pytest.mark.parametrize(
    "columns", [ncs.array(), "hello", nwp.nth(29).meta.as_selector(), ["a", "b", "A"]]
)
def test_unnest_frame_column_not_found_error(
    data: Data, columns: OneOrIterable[ColumnNameOrSelector], dataframe: DataFrame
) -> None:
    df = dataframe(data)
    with pytest.raises(ColumnNotFoundError):
        df.unnest(columns)


def test_unnest_frame_duplicate_error(data: Data, dataframe: DataFrame) -> None:
    df = dataframe(data).select(
        BEFORE,
        nwp.struct(AFTER, B, C).alias("one_conflict"),
        nwp.struct(BEFORE, C, D).alias("two_conflicts"),
        D,
        AFTER,
    )

    msg_one = r"'after'"
    msg_two = rf"'({D}|{BEFORE})'.+'({D}|{BEFORE})'"
    pattern_one = re_compile(msg_one)
    pattern_two = re_compile(msg_two)
    pattern_either = re_compile(rf"({msg_one})|({msg_two})")

    with pytest.raises(DuplicateError, match=pattern_one):
        df.unnest("one_conflict")
    with pytest.raises(DuplicateError, match=pattern_two):
        df.unnest("two_conflicts")
    with pytest.raises(DuplicateError, match=pattern_either):
        df.unnest(["one_conflict", "two_conflicts"])
    with pytest.raises(DuplicateError, match=pattern_either):
        df.unnest(ncs.struct())
