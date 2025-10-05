from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypedDict

import pytest

from narwhals.exceptions import NarwhalsError
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import TypeAlias

    from narwhals.typing import JoinStrategy
    from tests.conftest import Data

On: TypeAlias = "str | Sequence[str] | None"

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


class Keywords(TypedDict, total=False):
    """Arguments for `DataFrame.join`."""

    on: On
    how: JoinStrategy
    left_on: On
    right_on: On
    suffix: str


@pytest.fixture
def data_inner() -> Data:
    return {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zor ro": [7.0, 8.0, 9.0],
        "idx": [0, 1, 2],
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


@pytest.mark.xfail(
    reason=(
        "Did not raise on duplicate column names.\n"
        "Haven't added validation yet:\n"
        "https://github.com/narwhals-dev/narwhals/blob/f4787d3f9e027306cb1786db7b471f63b393b8d1/narwhals/_arrow/dataframe.py#L79-L93"
    )
)
def test_join_full_duplicate() -> None:
    df1 = {"foo": [1, 2, 3], "val1": [1, 2, 3]}
    df2 = {"foo": [1, 2, 3], "foo_right": [1, 2, 3]}
    df_left = dataframe(df1)
    df_right = dataframe(df2)

    with pytest.raises(NarwhalsError):
        df_left.join(df_right, "foo", how="full")


@pytest.mark.parametrize(
    "kwds",
    [
        Keywords(left_on="antananarivo", right_on="antananarivo"),
        Keywords(on="antananarivo"),
    ],
)
def test_join_inner_single_key(data_inner: Data, kwds: Keywords) -> None:
    df = dataframe(data_inner)
    result = df.join(df, **kwds).sort("idx").drop("idx_right")
    expected = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zor ro": [7.0, 8.0, 9.0],
        "idx": [0, 1, 2],
        "bob_right": [4, 4, 6],
        "zor ro_right": [7.0, 8.0, 9.0],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    "kwds",
    [
        Keywords(left_on=["antananarivo", "bob"], right_on=["antananarivo", "bob"]),
        Keywords(on=["antananarivo", "bob"]),
    ],
)
def test_join_inner_two_keys(data_inner: Data, kwds: Keywords) -> None:
    df = dataframe(data_inner)
    result = df.join(df, **kwds).sort("idx").drop("idx_right")
    expected = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zor ro": [7.0, 8.0, 9.0],
        "idx": [0, 1, 2],
        "zor ro_right": [7.0, 8.0, 9.0],
    }
    assert_equal_data(result, expected)


def test_join_left() -> None:
    data_left = {
        "antananarivo": [1.0, 2.0, 3.0],
        "bob": [4.0, 5.0, 6.0],
        "idx": [0.0, 1.0, 2.0],
    }
    data_right = {
        "antananarivo": [1.0, 2.0, 3.0],
        "co": [4.0, 5.0, 7.0],
        "idx": [0.0, 1.0, 2.0],
    }
    df_left = dataframe(data_left)
    df_right = dataframe(data_right)
    result = (
        df_left.join(df_right, left_on="bob", right_on="co", how="left")
        .sort("idx")
        .drop("idx_right")
    )
    expected = {
        "antananarivo": [1, 2, 3],
        "bob": [4, 5, 6],
        "idx": [0, 1, 2],
        "antananarivo_right": [1, 2, None],
    }
    result_on_list = df_left.join(df_right, ["antananarivo", "idx"], how="left").sort(
        "idx"
    )
    expected_on_list = {
        "antananarivo": [1, 2, 3],
        "bob": [4, 5, 6],
        "idx": [0, 1, 2],
        "co": [4, 5, 7],
    }
    assert_equal_data(result, expected)
    assert_equal_data(result_on_list, expected_on_list)


def test_join_left_multiple_column() -> None:
    data_left = {"antananarivo": [1, 2, 3], "bob": [4, 5, 6], "idx": [0, 1, 2]}
    data_right = {"antananarivo": [1, 2, 3], "c": [4, 5, 6], "idx": [0, 1, 2]}
    result = (
        dataframe(data_left)
        .join(
            dataframe(data_right),
            left_on=["antananarivo", "bob"],
            right_on=["antananarivo", "c"],
            how="left",
        )
        .sort("idx")
        .drop("idx_right")
    )
    expected = {"antananarivo": [1, 2, 3], "bob": [4, 5, 6], "idx": [0, 1, 2]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("kwds", "expected"),
    [
        (
            Keywords(left_on="bob", right_on="c"),
            {
                "antananarivo": [1, 2, 3],
                "bob": [4, 5, 6],
                "d": [1, 4, 2],
                "idx": [0, 1, 2],
                "antananarivo_right": [1, 2, 3],
                "d_right": [1, 4, 2],
            },
        ),
        (
            Keywords(left_on="antananarivo", right_on="d"),
            {
                "antananarivo": [1, 2, 3],
                "bob": [4, 5, 6],
                "d": [1, 4, 2],
                "idx": [0, 1, 2],
                "antananarivo_right": [1.0, 3.0, None],
                "c": [4.0, 6.0, None],
            },
        ),
    ],
)
def test_join_left_overlapping_column(kwds: Keywords, expected: dict[str, Any]) -> None:
    kwds["how"] = "left"
    data_left = {
        "antananarivo": [1.0, 2.0, 3.0],
        "bob": [4.0, 5.0, 6.0],
        "d": [1.0, 4.0, 2.0],
        "idx": [0.0, 1.0, 2.0],
    }
    data_right = {
        "antananarivo": [1.0, 2.0, 3.0],
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


@pytest.mark.xfail(
    reason=("Not implemented `how='cross'` yet"), raises=NotImplementedError
)
def test_join_cross() -> None:  # pragma: no cover
    df = dataframe({"a": [1, 3, 2]})
    result = df.join(df, how="cross").sort("a", "a_right")
    expected = {"a": [1, 1, 1, 2, 2, 2, 3, 3, 3], "a_right": [1, 2, 3, 1, 2, 3, 1, 2, 3]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("suffix", ["_right", "_custom_suffix"])
def test_join_with_suffix(how: JoinStrategy, suffix: str) -> None:
    data = {"a": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = dataframe(data)
    on = ["a", "bob"]
    result = df.join(df, left_on=on, right_on=on, how=how, suffix=suffix)
    assert result.schema.names() == ["a", "bob", "zor ro", f"zor ro{suffix}"]


@pytest.mark.xfail(
    reason=("Not implemented `how='cross'` yet"), raises=NotImplementedError
)
@pytest.mark.parametrize("suffix", ["_right", "_custom_suffix"])
def test_join_cross_with_suffix(suffix: str) -> None:  # pragma: no cover
    df = dataframe({"a": [1, 3, 2]})
    result = df.join(df, how="cross", suffix=suffix).sort("a", f"a{suffix}")
    expected = {
        "a": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        f"a{suffix}": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize("how", ["inner", "left", "semi", "anti"])
def test_join_keys_exceptions(how: JoinStrategy) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = dataframe(data)
    with pytest.raises(
        ValueError,
        match=rf"Either \(`left_on` and `right_on`\) or `on` keys should be specified for {how}.",
    ):
        df.join(df, how=how)
    with pytest.raises(
        ValueError,
        match=rf"Either \(`left_on` and `right_on`\) or `on` keys should be specified for {how}.",
    ):
        df.join(df, how=how, left_on="antananarivo")
    with pytest.raises(
        ValueError,
        match=rf"Either \(`left_on` and `right_on`\) or `on` keys should be specified for {how}.",
    ):
        df.join(df, how=how, right_on="antananarivo")
    with pytest.raises(
        ValueError,
        match=f"If `on` is specified, `left_on` and `right_on` should be None for {how}.",
    ):
        df.join(df, how=how, on="antananarivo", right_on="antananarivo")

    with pytest.raises(
        ValueError, match=re.escape("`left_on` and `right_on` must have the same length.")
    ):
        df.join(df, how=how, left_on=["antananarivo", "bob"], right_on="antananarivo")


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
    data = {"bob": [4, 4, 6]}
    df = dataframe(data)

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            f"Only the following join strategies are supported: ('inner', 'left', 'full', 'cross', 'semi', 'anti'); found '{how}'."
        ),
    ):
        df.join(df, left_on="bob", right_on="bob", how=how)  # type: ignore[arg-type]
