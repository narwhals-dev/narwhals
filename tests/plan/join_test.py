from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, TypeVar

import pytest

import narwhals._plan as nwp
from narwhals.exceptions import DuplicateError
from tests.plan.utils import assert_equal_data, dataframe, re_compile
from tests.utils import PYARROW_VERSION

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import TypeAlias

    from narwhals.typing import AsofJoinStrategy, JoinStrategy
    from tests.conftest import Data

By: TypeAlias = "str | Sequence[str] | None"
"""The type of `{by,by_left,by_right}`."""

OnT = TypeVar("OnT")
"""The type of `{on,left_on,right_on}`."""


class BaseKeywords(TypedDict, Generic[OnT], total=False):
    """Common arguments for `DataFrame.{join,join_asof}`."""

    on: OnT
    left_on: OnT
    right_on: OnT
    suffix: str


class AsofKwds(BaseKeywords["str | None"], total=False):
    """Arguments for `DataFrame.asof`."""

    by_left: By
    by_right: By
    by: By


class JoinKwds(BaseKeywords["str | Sequence[str] | None"], total=False):
    """Arguments for `DataFrame.join`."""

    how: JoinStrategy


@pytest.fixture
def data() -> Data:
    return {"a": [1, 3, 2], "b": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}


@pytest.fixture
def data_indexed(data: Data) -> Data:
    return data | {"idx": [0, 1, 2]}


@pytest.fixture
def data_a_only(data: Data) -> Data:
    return {"a": data["a"]}


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
            JoinKwds(left_on=["id"], right_on=["id"]),
        ),
        (LEFT_DATA_1, RIGHT_DATA_1, EXPECTED_DATA_1, JoinKwds(on="id")),
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
            JoinKwds(left_on=["id", "year"], right_on=["id", "year_foo"]),
        ),
    ],
    ids=["left_on-right_on-identical", "on", "left_on-right_on-different"],
)
def test_join_full(
    left_data: Data, right_data: Data, expected: Data, kwds: JoinKwds
) -> None:
    kwds["how"] = "full"
    result = (
        dataframe(left_data)
        .join(dataframe(right_data), **kwds)
        .sort("id", nulls_last=True)
    )
    assert_equal_data(result, expected)


def test_join_full_duplicate() -> None:
    left = dataframe({"f": [1, 2, 3], "v": [1, 2, 3]})
    right = left.rename({"v": "f_right"})
    with pytest.raises(DuplicateError):
        left.join(right, "f", how="full", suffix="_right")


def test_join_inner_x2_duplicate(data_indexed: Data) -> None:
    df = dataframe(data_indexed)
    with pytest.raises(DuplicateError):
        df.join(df, "a").join(df, "a")


@pytest.mark.parametrize("kwds", [JoinKwds(left_on="a", right_on="a"), JoinKwds(on="a")])
def test_join_inner_single_key(data_indexed: Data, kwds: JoinKwds) -> None:
    df = dataframe(data_indexed)
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
    "kwds", [JoinKwds(left_on=["a", "b"], right_on=["a", "b"]), JoinKwds(on=["a", "b"])]
)
def test_join_inner_two_keys(data_indexed: Data, kwds: JoinKwds) -> None:
    df = dataframe(data_indexed)
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
    df = dataframe({"a": [1, 2, 3], "b": [4, 5, 6], "idx": [0, 1, 2]})
    right = df.rename({"b": "c"})
    result = (
        df.join(right, left_on=["a", "b"], right_on=["a", "c"], how="left")
        .sort("idx")
        .drop("idx_right")
    )
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "idx": [0, 1, 2]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("kwds", "expected"),
    [
        (
            JoinKwds(left_on="b", right_on="c"),
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
            JoinKwds(left_on="a", right_on="d"),
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
def test_join_left_overlapping_column(kwds: JoinKwds, expected: dict[str, Any]) -> None:
    kwds["how"] = "left"
    source = {
        "a": [1.0, 2.0, 3.0],
        "b": [4.0, 5.0, 6.0],
        "d": [1.0, 4.0, 2.0],
        "idx": [0.0, 1.0, 2.0],
    }
    df = dataframe(source)
    right = df.rename({"b": "c"})
    result = df.join(right, **kwds).sort("idx").drop("idx_right")
    assert_equal_data(result, expected)


def test_join_cross(data_a_only: Data) -> None:
    df = dataframe(data_a_only)
    result = df.join(df, how="cross").sort("a", "a_right")
    expected = {"a": [1, 1, 1, 2, 2, 2, 3, 3, 3], "a_right": [1, 2, 3, 1, 2, 3, 1, 2, 3]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("suffix", ["_right", "_custom_suffix"])
def test_join_with_suffix(how: JoinStrategy, suffix: str, data: Data) -> None:
    df = dataframe(data)
    on = ["a", "b"]
    result = df.join(df, left_on=on, right_on=on, how=how, suffix=suffix)
    assert result.schema.names() == ["a", "b", "zor ro", f"zor ro{suffix}"]


@pytest.mark.parametrize("suffix", ["_right", "_custom_suffix"])
def test_join_cross_with_suffix(suffix: str, data_a_only: Data) -> None:
    df = dataframe(data_a_only)
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
    data: Data,
) -> None:
    # NOTE: "anti" and "semi" should be the inverse of each other
    df = dataframe(data)
    other = df.filter(predicate if how == "semi" else ~predicate)
    result = df.join(other, on, how=how).sort(on)
    assert_equal_data(result, expected)


EITHER_LR_OR_ON = r"`left_on` and `right_on`.+or.+`on`"
ONLY_ON = r"`on` is specified.+`left_on` and `right_on`.+be.+None"
SAME_LENGTH_ON = r"`left_on` and `right_on`.+same length"


@pytest.mark.parametrize(
    ("kwds", "message"),
    [
        (JoinKwds(), EITHER_LR_OR_ON),
        (JoinKwds(left_on="a"), EITHER_LR_OR_ON),
        (JoinKwds(right_on="a"), EITHER_LR_OR_ON),
        (JoinKwds(on="a", right_on="a"), ONLY_ON),
        (JoinKwds(left_on=["a", "b"], right_on="a"), SAME_LENGTH_ON),
    ],
)
@pytest.mark.parametrize("how", ["inner", "left", "semi", "anti"])
def test_join_keys_exceptions(
    how: JoinStrategy, kwds: JoinKwds, message: str, data: Data
) -> None:
    df = dataframe(data)
    kwds["how"] = how
    with pytest.raises(ValueError, match=message):
        df.join(df, **kwds)


@pytest.mark.parametrize(
    "kwds",
    [
        JoinKwds(left_on="a"),
        JoinKwds(on="a"),
        JoinKwds(right_on="a"),
        JoinKwds(left_on="a", right_on="a"),
    ],
)
def test_join_cross_keys_exceptions(kwds: JoinKwds, data_a_only: Data) -> None:
    df = dataframe(data_a_only)
    kwds["how"] = "cross"
    with pytest.raises(ValueError, match=r"not.+ `left_on`.+`right_on`.+`on`.+cross"):
        df.join(df, **kwds)


def test_join_not_implemented(data_a_only: Data) -> None:
    df = dataframe(data_a_only)
    pattern = (
        r"supported.+'inner', 'left', 'full', 'cross', 'semi', 'anti'.+ found 'right'"
    )
    with pytest.raises(NotImplementedError, match=(pattern)):
        df.join(df, left_on="a", right_on="a", how="right")  # type: ignore[arg-type]


# NOTE: move `join_asof` to a different file later


PYARROW_HAS_JOIN_ASOF = PYARROW_VERSION >= (16, 0, 0)


def require_pyarrow_16(
    df: nwp.DataFrame[Any, Any], request: pytest.FixtureRequest
) -> None:
    request.applymarker(
        pytest.mark.xfail(
            (df.implementation.is_pyarrow() and not PYARROW_HAS_JOIN_ASOF),
            reason="pyarrow too old for `join_asof` support",
            raises=NotImplementedError,
        )
    )


XFAIL_NEAREST = pytest.mark.xfail(
    PYARROW_HAS_JOIN_ASOF,
    reason="Only 'backward' and 'forward' strategies are currently supported for `pyarrow`",
    raises=NotImplementedError,
)

XFAIL_SUFFIX = pytest.mark.xfail(
    (PYARROW_HAS_JOIN_ASOF),
    reason="`pyarrow.acero.AsofJoinNodeOptions` does not support column collisions.",
    raises=ValueError,
)

# TODO @dangotbanned: Why is `left_by=[], right_by=[]` giving no matches?
# https://github.com/apache/arrow/blob/9b03118e834dfdaa0cf9e03595477b499252a9cb/python/pyarrow/tests/test_table.py#L3238-L3255
XFAIL_EMPTY_BY = pytest.mark.xfail(
    PYARROW_HAS_JOIN_ASOF, reason="Empty `by` fails to match?", raises=AssertionError
)


@XFAIL_SUFFIX
@pytest.mark.parametrize(
    ("strategy", "expected_values"),
    [
        ("backward", [1, 3, 7]),
        ("forward", [1, 6, None]),
        pytest.param("nearest", [1, 6, 7], marks=XFAIL_NEAREST),
    ],
    ids=str,
)
@pytest.mark.parametrize("kwds", [AsofKwds(left_on="a", right_on="a"), AsofKwds(on="a")])
def test_join_asof_numeric(
    strategy: AsofJoinStrategy,
    expected_values: list[Any],
    request: pytest.FixtureRequest,
    kwds: AsofKwds,
) -> None:
    left = {"a": [1, 5, 10], "val": ["a", "b", "c"]}
    right = {"a": [1, 2, 3, 6, 7], "val": [1, 2, 3, 6, 7]}
    expected = left | {"val_right": expected_values}
    df = dataframe(left).sort("a")
    require_pyarrow_16(df, request)
    df_right = dataframe(right).sort("a")
    result = df.join_asof(df_right, **kwds, strategy=strategy).sort("a")
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("strategy", "expected_values"),
    [
        ("backward", [4164, 4566, 4696]),
        ("forward", [4411, 4696, 4696]),
        pytest.param("nearest", [4164, 4696, 4696], marks=XFAIL_NEAREST),
    ],
    ids=str,
)
@pytest.mark.parametrize(
    "kwds", [AsofKwds(left_on="ts", right_on="ts"), AsofKwds(on="ts")]
)
def test_join_asof_time(
    strategy: AsofJoinStrategy,
    expected_values: list[float],
    request: pytest.FixtureRequest,
    kwds: AsofKwds,
) -> None:
    left = {
        "ts": [dt.datetime(2016, 3, 1), dt.datetime(2018, 8, 1), dt.datetime(2019, 1, 1)],
        "pop": [82.19, 82.66, 83.12],
    }
    right = {
        "ts": [
            dt.datetime(2016, 1, 1),
            dt.datetime(2017, 1, 1),
            dt.datetime(2018, 1, 1),
            dt.datetime(2019, 1, 1),
            dt.datetime(2020, 1, 1),
        ],
        "gdp": [4164, 4411, 4566, 4696, 4827],
    }
    expected = left | {"gdp": expected_values}
    df = dataframe(left).sort("ts")
    require_pyarrow_16(df, request)
    df_right = dataframe(right).sort("ts")
    result = df.join_asof(df_right, **kwds, strategy=strategy).sort("ts")
    request.applymarker(XFAIL_EMPTY_BY)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    "kwds", [AsofKwds(on="a", by_left="b", by_right="b"), AsofKwds(on="a", by="b")]
)
def test_join_asof_by(request: pytest.FixtureRequest, kwds: AsofKwds) -> None:
    left = {"a": [1, 5, 7, 10], "b": ["D", "D", "C", "A"], "c": [9, 2, 1, 1]}
    right = {"a": [1, 4, 5, 8], "b": ["D", "D", "A", "F"], "d": [1, 3, 4, 1]}
    expected = {
        "a": [1, 5, 7, 10],
        "b": ["D", "D", "C", "A"],
        "c": [9, 2, 1, 1],
        "d": [1, 3, None, 4],
    }
    df = dataframe(left).sort("a")
    require_pyarrow_16(df, request)
    df_right = dataframe(right).sort("a")
    result = df.join_asof(df_right, **kwds).sort("a")
    assert_equal_data(result, expected)


@XFAIL_SUFFIX
@pytest.mark.parametrize("kwds", [AsofKwds(left_on="a", right_on="a", suffix="_y")])
def test_join_asof_suffix(request: pytest.FixtureRequest, kwds: AsofKwds) -> None:
    left = {"a": [1, 5, 10], "val": ["a", "b", "c"]}
    right = {"a": [1, 2, 3, 6, 7], "val": [1, 2, 3, 6, 7]}
    expected = {"a": [1, 5, 10], "val": ["a", "b", "c"], "val_y": [1, 3, 7]}
    df = dataframe(left).sort("a")
    require_pyarrow_16(df, request)
    df_right = dataframe(right).sort("a")
    result = df.join_asof(df_right, **kwds).sort("a")
    assert_equal_data(result, expected)


@pytest.mark.parametrize("strategy", ["back", "furthest"])
def test_join_asof_not_implemented(strategy: str, data: Data) -> None:
    df = dataframe(data)
    pattern = re_compile(
        rf"supported.+'backward', 'forward', 'nearest'.+ found '{strategy}'"
    )
    with pytest.raises(NotImplementedError, match=pattern):
        df.join_asof(df, left_on="a", right_on="a", strategy=strategy)  # type: ignore[arg-type]


EITHER_LR_OR_BY = r"If.+by.+by_left.+by_right.+should be None"
SAME_LENGTH_BY = r"by_left.+by_right.+same.+length"
BOTH_BY = r"not.+by_left.+or.+by_right.+need.+both"
ON = "a"
BY = "b"


@pytest.mark.parametrize(
    ("kwds", "message"),
    [
        (AsofKwds(), EITHER_LR_OR_ON),
        (AsofKwds(left_on=ON), EITHER_LR_OR_ON),
        (AsofKwds(right_on=ON), EITHER_LR_OR_ON),
        (AsofKwds(on=ON, right_on=ON), ONLY_ON),
        (AsofKwds(on=ON, left_on=ON, right_on=ON), ONLY_ON),
        (AsofKwds(on=ON, left_on=ON), ONLY_ON),
        (AsofKwds(on=ON, by=BY, by_left=BY, by_right=BY), EITHER_LR_OR_BY),
        (AsofKwds(on=ON, by=BY, by_left=BY), EITHER_LR_OR_BY),
        (AsofKwds(on=ON, by=BY, by_right=BY), EITHER_LR_OR_BY),
        (AsofKwds(on=ON, by_left=[ON, BY], by_right=BY), SAME_LENGTH_BY),
        (AsofKwds(on=ON, by_left=BY), BOTH_BY),
        (AsofKwds(on=ON, by_right=BY), BOTH_BY),
    ],
)
def test_join_asof_invalid(data: Data, kwds: AsofKwds, message: str) -> None:
    df = dataframe(data)
    with pytest.raises(ValueError, match=message):
        df.join_asof(df, **kwds)
