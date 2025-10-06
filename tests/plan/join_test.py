from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import pytest

import narwhals._plan as nwp
from narwhals.exceptions import DuplicateError, NarwhalsError
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import TypeAlias

    from narwhals.typing import JoinStrategy
    from tests.conftest import Data

    On: TypeAlias = "str | Sequence[str] | None"


class Keywords(TypedDict, total=False):
    """Arguments for `DataFrame.join`."""

    on: On
    how: JoinStrategy
    left_on: On
    right_on: On
    suffix: str


XFAIL_DUPLICATE_COLUMN_NAMES = pytest.mark.xfail(
    reason=(
        "Did not raise on duplicate column names.\n"
        "Haven't added validation yet:\n"
        "https://github.com/narwhals-dev/narwhals/blob/f4787d3f9e027306cb1786db7b471f63b393b8d1/narwhals/_arrow/dataframe.py#L79-L93"
    )
)


@pytest.fixture
def data_inner() -> Data:
    return {"a": [1, 3, 2], "b": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0], "idx": [0, 1, 2]}


LEFT_DATA_1 = {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
RIGHT_DATA_1 = {
    "id": [2, 3, 4],
    "department": ["HR", "Engineering", "Marketing"],
    "salary": [50000, 60000, 70000],
}
EXPECTED_DATA_1 = {
    "id": [1, 2, 3, None],
    "name": ["Alice", "Bob", "Charlie", None],
    "age": [25, 30, 35, None],
    "id_right": [None, 2, 3, 4],
    "department": [None, "HR", "Engineering", "Marketing"],
    "salary": [None, 50000, 60000, 70000],
}


@pytest.mark.parametrize(
    ("left_data", "right_data", "expected", "kwds"),
    [
        (
            LEFT_DATA_1,
            RIGHT_DATA_1,
            EXPECTED_DATA_1,
            Keywords(left_on=["id"], right_on=["id"]),
        ),
        (LEFT_DATA_1, RIGHT_DATA_1, EXPECTED_DATA_1, Keywords(on="id")),
        (
            {
                "id": [1, 2, 3, 4],
                "year": [2020, 2021, 2022, 2023],
                "value1": [100, 200, 300, 400],
            },
            {
                "id": [2, 3, 4, 5],
                "year_foo": [2021, 2022, 2023, 2024],
                "value2": [500, 600, 700, 800],
            },
            {
                "id": [1, 2, 3, 4, None],
                "year": [2020, 2021, 2022, 2023, None],
                "value1": [100, 200, 300, 400, None],
                "id_right": [None, 2, 3, 4, 5],
                # since year is different, don't apply suffix
                "year_foo": [None, 2021, 2022, 2023, 2024],
                "value2": [None, 500, 600, 700, 800],
            },
            Keywords(left_on=["id", "year"], right_on=["id", "year_foo"]),
        ),
    ],
    ids=["left_on-right_on-identical", "on", "left_on-right_on-different"],
)
def test_join_full(
    left_data: Data, right_data: Data, expected: Data, kwds: Keywords
) -> None:
    kwds["how"] = "full"
    result = (
        dataframe(left_data)
        .join(dataframe(right_data), **kwds)
        .sort("id", nulls_last=True)
    )
    assert_equal_data(result, expected)


@XFAIL_DUPLICATE_COLUMN_NAMES
def test_join_full_duplicate() -> None:
    df1 = {"foo": [1, 2, 3], "val1": [1, 2, 3]}
    df2 = {"foo": [1, 2, 3], "foo_right": [1, 2, 3]}
    df_left = dataframe(df1)
    df_right = dataframe(df2)

    with pytest.raises(NarwhalsError):
        df_left.join(df_right, "foo", how="full")


@XFAIL_DUPLICATE_COLUMN_NAMES
def test_join_duplicate_column_names() -> None:
    data = {"a": [1, 2, 3, 4, 5], "b": [6, 6, 6, 6, 6]}
    df = dataframe(data)
    with pytest.raises(DuplicateError):
        df.join(df, "a").join(df, "a")


@pytest.mark.parametrize("kwds", [Keywords(left_on="a", right_on="a"), Keywords(on="a")])
def test_join_inner_single_key(data_inner: Data, kwds: Keywords) -> None:
    df = dataframe(data_inner)
    result = df.join(df, **kwds).sort("idx").drop("idx_right")
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "zor ro": [7.0, 8.0, 9.0],
        "idx": [0, 1, 2],
        "b_right": [4, 4, 6],
        "zor ro_right": [7.0, 8.0, 9.0],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    "kwds", [Keywords(left_on=["a", "b"], right_on=["a", "b"]), Keywords(on=["a", "b"])]
)
def test_join_inner_two_keys(data_inner: Data, kwds: Keywords) -> None:
    df = dataframe(data_inner)
    result = df.join(df, **kwds).sort("idx").drop("idx_right")
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "zor ro": [7.0, 8.0, 9.0],
        "idx": [0, 1, 2],
        "zor ro_right": [7.0, 8.0, 9.0],
    }
    assert_equal_data(result, expected)


def test_join_left() -> None:
    data_left = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "idx": [0.0, 1.0, 2.0]}
    data_right = {"a": [1.0, 2.0, 3.0], "co": [4.0, 5.0, 7.0], "idx": [0.0, 1.0, 2.0]}
    df_left = dataframe(data_left)
    df_right = dataframe(data_right)
    result = (
        df_left.join(df_right, left_on="b", right_on="co", how="left")
        .sort("idx")
        .drop("idx_right")
    )
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "idx": [0, 1, 2], "a_right": [1, 2, None]}
    result_on_list = df_left.join(df_right, ["a", "idx"], how="left").sort("idx")
    expected_on_list = {"a": [1, 2, 3], "b": [4, 5, 6], "idx": [0, 1, 2], "co": [4, 5, 7]}
    assert_equal_data(result, expected)
    assert_equal_data(result_on_list, expected_on_list)


def test_join_left_multiple_column() -> None:
    data_left = {"a": [1, 2, 3], "b": [4, 5, 6], "idx": [0, 1, 2]}
    data_right = {"a": [1, 2, 3], "c": [4, 5, 6], "idx": [0, 1, 2]}
    result = (
        dataframe(data_left)
        .join(dataframe(data_right), left_on=["a", "b"], right_on=["a", "c"], how="left")
        .sort("idx")
        .drop("idx_right")
    )
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "idx": [0, 1, 2]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("kwds", "expected"),
    [
        (
            Keywords(left_on="b", right_on="c"),
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "d": [1, 4, 2],
                "idx": [0, 1, 2],
                "a_right": [1, 2, 3],
                "d_right": [1, 4, 2],
            },
        ),
        (
            Keywords(left_on="a", right_on="d"),
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "d": [1, 4, 2],
                "idx": [0, 1, 2],
                "a_right": [1.0, 3.0, None],
                "c": [4.0, 6.0, None],
            },
        ),
    ],
)
def test_join_left_overlapping_column(kwds: Keywords, expected: dict[str, Any]) -> None:
    kwds["how"] = "left"
    data_left = {
        "a": [1.0, 2.0, 3.0],
        "b": [4.0, 5.0, 6.0],
        "d": [1.0, 4.0, 2.0],
        "idx": [0.0, 1.0, 2.0],
    }
    data_right = {
        "a": [1.0, 2.0, 3.0],
        "c": [4.0, 5.0, 6.0],
        "d": [1.0, 4.0, 2.0],
        "idx": [0.0, 1.0, 2.0],
    }
    result = (
        dataframe(data_left)
        .join(dataframe(data_right), **kwds)
        .sort("idx")
        .drop("idx_right")
    )
    assert_equal_data(result, expected)


def test_join_cross() -> None:
    df = dataframe({"a": [1, 3, 2]})
    result = df.join(df, how="cross").sort("a", "a_right")
    expected = {"a": [1, 1, 1, 2, 2, 2, 3, 3, 3], "a_right": [1, 2, 3, 1, 2, 3, 1, 2, 3]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("suffix", ["_right", "_custom_suffix"])
def test_join_with_suffix(how: JoinStrategy, suffix: str) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = dataframe(data)
    on = ["a", "b"]
    result = df.join(df, left_on=on, right_on=on, how=how, suffix=suffix)
    assert result.schema.names() == ["a", "b", "zor ro", f"zor ro{suffix}"]


@pytest.mark.parametrize("suffix", ["_right", "_custom_suffix"])
def test_join_cross_with_suffix(suffix: str) -> None:
    df = dataframe({"a": [1, 3, 2]})
    result = df.join(df, how="cross", suffix=suffix).sort("a", f"a{suffix}")
    expected = {
        "a": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        f"a{suffix}": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("on", "predicate", "expected"),
    [
        (["a", "b"], (nwp.col("b") < 5), {"a": [1, 3], "b": [4, 4], "zor ro": [7, 8]}),
        (["b"], (nwp.col("b") < 5), {"a": [1, 3], "b": [4, 4], "zor ro": [7, 8]}),
        (["b"], (nwp.col("b") > 5), {"a": [2], "b": [6], "zor ro": [9]}),
    ],
)
@pytest.mark.parametrize("how", ["anti", "semi"])
def test_join_filter(
    on: str | Sequence[str],
    predicate: nwp.Expr,
    how: Literal["anti", "semi"],
    expected: Data,
) -> None:
    # NOTE: "anti" and "semi" should be the inverse of eachother
    df = dataframe({"a": [1, 3, 2], "b": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]})
    other = df.filter(predicate if how == "semi" else ~predicate)
    result = df.join(other, on, how=how).sort(on)
    assert_equal_data(result, expected)


EITHER_LR_OR_ON = r"`left_on` and `right_on`.+or.+`on`"
ONLY_ON = r"`on` is specified.+`left_on` and `right_on`.+be.+None"
SAME_LENGTH = r"`left_on` and `right_on`.+same length"


@pytest.mark.parametrize(
    ("kwds", "message"),
    [
        (Keywords(), EITHER_LR_OR_ON),
        (Keywords(left_on="a"), EITHER_LR_OR_ON),
        (Keywords(right_on="a"), EITHER_LR_OR_ON),
        (Keywords(on="a", right_on="a"), ONLY_ON),
        (Keywords(left_on=["a", "b"], right_on="a"), SAME_LENGTH),
    ],
)
@pytest.mark.parametrize("how", ["inner", "left", "semi", "anti"])
def test_join_keys_exceptions(how: JoinStrategy, kwds: Keywords, message: str) -> None:
    df = dataframe({"a": [1], "b": [4]})
    kwds["how"] = how
    with pytest.raises(ValueError, match=message):
        df.join(df, **kwds)


@pytest.mark.parametrize(
    "kwds",
    [
        Keywords(left_on="a"),
        Keywords(on="a"),
        Keywords(right_on="a"),
        Keywords(left_on="a", right_on="a"),
    ],
)
def test_join_cross_keys_exceptions(kwds: Keywords) -> None:
    df = dataframe({"a": [1, 3, 2]})
    kwds["how"] = "cross"
    with pytest.raises(
        ValueError, match="Can not pass `left_on`, `right_on` or `on` keys for cross join"
    ):
        df.join(df, **kwds)


@pytest.mark.parametrize("how", ["right"])
def test_join_not_implemented(how: str) -> None:
    data = {"b": [4, 4, 6]}
    df = dataframe(data)

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            f"Only the following join strategies are supported: ('inner', 'left', 'full', 'cross', 'semi', 'anti'); found '{how}'."
        ),
    ):
        df.join(df, left_on="b", right_on="b", how=how)  # type: ignore[arg-type]
