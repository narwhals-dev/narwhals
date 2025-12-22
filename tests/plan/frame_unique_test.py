from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pytest

from narwhals import _plan as nwp
from narwhals._plan import selectors as ncs
from narwhals.exceptions import ColumnNotFoundError, InvalidOperationError
from tests.plan.utils import assert_equal_data, dataframe, re_compile

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from tests.conftest import Data

    OrderedStrategy: TypeAlias = Literal["first", "last"]
    UnorderedStrategy: TypeAlias = Literal["any", "none"]


# TODO @dangotbanned: Either, define a more complete `data` which every test uses
# or define multiple but reuse them
# Currently 4 inline `data`s + the fixture
@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


def test_unique_invalid_keep(data: Data) -> None:
    with pytest.raises(NotImplementedError, match=re_compile(r"found.+cabbage")):
        dataframe(data).unique(keep="cabbage")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("subset", "err"),
    [
        ("cabbage", ColumnNotFoundError),
        (ncs.string(), ColumnNotFoundError),
        (nwp.nth(5).meta.as_selector(), ColumnNotFoundError),
        (["a", "b", "A"], ColumnNotFoundError),
        (nwp.col("a").first(), InvalidOperationError),
        pytest.param(
            ncs.first().last(),
            InvalidOperationError,
            # TODO @dangotbanned: Fix this in another PR
            # Need to be stricter on the Selector check
            marks=pytest.mark.xfail(
                reason="narwhals/_plan/_expansion.py:160: 'Last' object has no attribute 'iter_expand_names'",
                raises=AttributeError,
            ),
        ),
    ],
)
def test_unique_invalid_subset(data: Data, subset: Any, err: type[Exception]) -> None:
    df = dataframe(data)
    with pytest.raises(err):
        df.unique(subset)


@pytest.mark.parametrize("subset", ["b", ["b"]])
@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("first", {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}),
        ("last", {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}),
    ],
)
def test_unique_eager(
    data: Data,
    subset: str | list[str] | None,
    keep: OrderedStrategy,
    expected: dict[str, list[float]],
) -> None:
    result = dataframe(data).unique(subset, keep=keep).sort("z")
    assert_equal_data(result, expected)


# TODO @dangotbanned: Handle these cases more cleanly
# (all) `unique(...).sort()` vs
# (eager only) `unique(..., maintain_order=True)`
@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("first", {"i": [None, 2], "a": [2, 1], "b": [4, 6]}),
        ("last", {"i": [1, 2], "a": [3, 1], "b": [4, 6]}),
    ],
)
def test_unique_first_last(
    keep: OrderedStrategy, expected: dict[str, list[float]]
) -> None:
    data = {"i": [0, 1, None, 2], "a": [1, 3, 2, 1], "b": [4, 4, 4, 6]}
    df = dataframe(data)
    result = df.unique("b", keep=keep, order_by="i").sort("i")
    assert_equal_data(result, expected)
    result = df.unique("b", keep=keep, order_by="i", maintain_order=True)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("first", {"i": [0, 1, 2], "b": [4, 4, 6]}),
        ("last", {"i": [0, 1, 2], "b": [4, 4, 6]}),
    ],
)
def test_unique_first_last_no_subset(
    keep: OrderedStrategy, expected: dict[str, list[float]]
) -> None:
    data = {"i": [0, 1, 1, 2], "b": [4, 4, 4, 6]}
    df = dataframe(data)
    result = df.unique(keep=keep, order_by="i").sort("i")
    assert_equal_data(result, expected)
    result = df.unique(keep=keep, order_by="i", maintain_order=True)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("any", {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}),
        ("none", {"a": [2], "b": [6], "z": [9]}),
    ],
)
def test_unique(
    data: Data, keep: UnorderedStrategy, expected: dict[str, list[float]]
) -> None:
    result = dataframe(data).unique(["b"], keep=keep).sort("z")
    assert_equal_data(result, expected)


@pytest.mark.parametrize("subset", [None, ["a", "b"]])
@pytest.mark.parametrize(
    ("keep", "expected"),
    [("any", {"a": [1, 1, 2], "b": [3, 4, 4]}), ("none", {"a": [1, 2], "b": [4, 4]})],
)
def test_unique_full_subset(
    subset: list[str] | None, keep: UnorderedStrategy, expected: dict[str, list[float]]
) -> None:
    data = {"a": [1, 1, 1, 2], "b": [3, 3, 4, 4]}
    result = dataframe(data).unique(subset, keep=keep).sort("a", "b")
    assert_equal_data(result, expected)


def test_unique_none(data: Data) -> None:
    df = dataframe(data)
    result = df.unique().sort("z")
    assert_equal_data(result, data)
    result = df.unique(maintain_order=True)
    assert_equal_data(result, data)


def test_unique_3069() -> None:
    data = {"name": ["a", "b", "c"], "group": ["d", "e", "f"], "value": [1, 2, 3]}
    group = ncs.by_name("group")
    result = dataframe(data).select(group).unique().sort(group)
    expected = {"group": ["d", "e", "f"]}
    assert_equal_data(result, expected)
