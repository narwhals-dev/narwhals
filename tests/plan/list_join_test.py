from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from typing import Final, TypeVar

    from typing_extensions import TypeAlias

    from tests.conftest import Data

    T = TypeVar("T")
    SubList: TypeAlias = list[T] | list[T | None] | list[None] | None
    SubListStr: TypeAlias = SubList[str]


R1: Final[SubListStr] = ["a", "b", "c"]
R2: Final[SubListStr] = [None, None, None]
R3: Final[SubListStr] = [None, None, "1", "2", None, "3", None]
R4: Final[SubListStr] = ["x", "y"]
R5: Final[SubListStr] = ["1", None, "3"]
R6: Final[SubListStr] = [None]
R7: Final[SubListStr] = None
R8: Final[SubListStr] = []
R9: Final[SubListStr] = [None, None]


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [R1, R2, R3, R4, R5, R6, R7, R8, R9]}


a = nwp.col("a")


@pytest.mark.parametrize(
    ("separator", "ignore_nulls", "expected"),
    [
        ("-", False, ["a-b-c", None, None, "x-y", None, None, None, "", None]),
        ("-", True, ["a-b-c", "", "1-2-3", "x-y", "1-3", "", None, "", ""]),
        ("", False, ["abc", None, None, "xy", None, None, None, "", None]),
        ("", True, ["abc", "", "123", "xy", "13", "", None, "", ""]),
    ],
    ids=[
        "hyphen-propagate_nulls",
        "hyphen-ignore_nulls",
        "empty-propagate_nulls",
        "empty-ignore_nulls",
    ],
)
def test_list_join(
    data: Data, separator: str, *, ignore_nulls: bool, expected: list[str | None]
) -> None:
    df = dataframe(data).with_columns(a.cast(nw.List(nw.String)))
    expr = a.list.join(separator, ignore_nulls=ignore_nulls)
    result = df.select(expr)
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    "ignore_nulls", [True, False], ids=["ignore_nulls", "propagate_nulls"]
)
@pytest.mark.parametrize("separator", ["?", "", " "], ids=["question", "empty", "space"])
@pytest.mark.parametrize(
    "row", [R1, R2, R3, R4, R5, R6, R7, R8, R9], ids=[f"row-{i}" for i in range(1, 10)]
)
def test_list_join_scalar(row: SubListStr, separator: str, *, ignore_nulls: bool) -> None:
    data = {"a": [row]}
    df = dataframe(data).select(a.cast(nw.List(nw.String)))
    expr = a.first().list.join(separator, ignore_nulls=ignore_nulls)
    result = df.select(expr)
    expected: str | None
    if row is None:
        expected = None
    elif row == []:
        expected = ""
    elif any(el is None for el in row):
        if not ignore_nulls:
            expected = None
        elif all(el is None for el in row):
            expected = ""
        else:
            expected = separator.join(el for el in row if el is not None)
    else:
        expected = separator.join(el for el in row if el is not None)

    assert_equal_data(result, {"a": [expected]})
