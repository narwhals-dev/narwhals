from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

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


@pytest.mark.parametrize(
    ("left_data", "right_data", "expected", "on", "left_on", "right_on"),
    [
        (LEFT_DATA_1, RIGHT_DATA_1, EXPECTED_DATA_1, None, ["id"], ["id"]),
        (LEFT_DATA_1, RIGHT_DATA_1, EXPECTED_DATA_1, "id", None, None),
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
            None,
            ["id", "year"],
            ["id", "year_foo"],
        ),
    ],
    ids=["left_on-right_on-identical", "on", "left_on-right_on-different"],
)
def test_join_full(
    left_data: Data, right_data: Data, expected: Data, on: On, left_on: On, right_on: On
) -> None:
    result = (
        dataframe(left_data)
        .join(
            dataframe(right_data), on=on, left_on=left_on, right_on=right_on, how="full"
        )
        .sort("id", nulls_last=True)
    )
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
