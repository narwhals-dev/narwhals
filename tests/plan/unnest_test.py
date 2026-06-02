from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal

import pytest

import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from narwhals.exceptions import ColumnNotFoundError, DuplicateError, InvalidOperationError
from tests.plan.utils import DataFrame, assert_equal_data, re_compile

if TYPE_CHECKING:
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


# TODO @dangotbanned: Rewrite this whole thing (using indices was *because of* pyarrow's api)
@pytest.mark.parametrize("insert_at", [-1, 0, 1])
@pytest.mark.parametrize(
    "columns", [[A], [A, B], [A, B, C, D]], ids=["1-column", "2-column", "4-column"]
)
def test_unnest_frame_single_struct(
    data: Data, columns: list[str], insert_at: Literal[-1, 0, 1], dataframe: DataFrame
) -> None:
    expected = copy.deepcopy(data)
    if insert_at in {-1, 0}:
        if insert_at == -1:
            for column in columns:
                expected[column] = expected.pop(column)
        else:
            _tmp: Data = {}
            for column in columns:
                _tmp[column] = expected.pop(column)
            _tmp |= expected
            expected = _tmp

    df = dataframe(data)
    struct_name = "t_struct"
    struct = nwp.struct(columns).alias("t_struct")

    df = df.with_columns(struct).drop(columns)
    if insert_at == 0:
        df = df.select(struct_name, nwp.exclude(struct_name))
    elif insert_at == -1:
        df = df.select(nwp.exclude(struct_name), struct_name)
    else:
        df = df.select(ncs.first(), struct_name, ~(ncs.first() | ncs.last()))

    assert_equal_data(df.unnest("t_struct"), expected)
    assert_equal_data(df.unnest(ncs.struct()), expected)
    assert_equal_data(df.unnest(nwp.nth(insert_at).meta.as_selector()), expected)


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
    assert_equal_data(df.unnest(ncs.by_index(2, 1)), expected_reorder)


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
