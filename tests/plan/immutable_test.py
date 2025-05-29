from __future__ import annotations

from typing import Any

import pytest

from narwhals._plan.common import Immutable


class Empty(Immutable): ...


class EmptyDerived(Empty):
    __slots__ = ("a",)
    a: int


class OneSlot(Immutable):
    __slots__ = ("a",)
    a: int


class TwoSlot(Immutable):
    __slots__ = ("a", "b")
    a: int
    b: str


@pytest.fixture
def empty() -> Empty:
    return Empty()


@pytest.fixture
def empty_derived() -> EmptyDerived:
    return EmptyDerived(a=1)


@pytest.fixture
def one() -> OneSlot:
    return OneSlot(a=1)


@pytest.fixture
def two() -> TwoSlot:
    return TwoSlot(a=1, b="two")


def test_immutable_really_immutable(
    empty: Empty, empty_derived: EmptyDerived, one: OneSlot, two: TwoSlot
) -> None:
    with pytest.raises(AttributeError, match=r"Empty.+immutable.+'a'"):
        empty.a = 1  # type: ignore[assignment]
    assert empty_derived.a == 1
    with pytest.raises(AttributeError, match=r"EmptyDerived.+immutable.+'a'"):
        empty_derived.a = 2  # type: ignore[misc]
    with pytest.raises(AttributeError, match=r"OneSlot.+immutable.+'a'"):
        one.a = 2  # type: ignore[misc]
    with pytest.raises(AttributeError, match=r"OneSlot.+immutable.+'b'"):
        one.b = "two"  # type: ignore[assignment]
    with pytest.raises(AttributeError, match=r"TwoSlot.+immutable.+'a'"):
        two.a += 2  # type: ignore[misc]
    with pytest.raises(AttributeError, match=r"TwoSlot.+immutable.+'b'"):
        two.b = 2  # type: ignore[assignment, misc]


def test_immutable_hash(
    empty: Empty, empty_derived: EmptyDerived, one: OneSlot, two: TwoSlot
) -> None:
    class EmptyAgain(Immutable): ...

    assert empty == Empty()
    assert empty_derived == EmptyDerived(a=1)
    assert one == OneSlot(a=1)
    assert two == TwoSlot(a=1, b="two")

    assert empty_derived != EmptyDerived(a=2)
    assert one != OneSlot(a=2)
    assert two != TwoSlot(a=2, b="two")
    assert two != TwoSlot(a=1, b="three")
    assert two != TwoSlot(a=2, b="three")

    assert empty != empty_derived
    assert empty_derived != one
    assert one != two
    empty_again = EmptyAgain()
    assert empty != empty_again

    mapping: dict[Any, Any] = {empty: empty}
    mapping.update([(empty_derived, empty_derived), (one, one), (two, two)])
    assert len(mapping) == 4
    mapping[empty_again] = empty_again
    assert len(mapping) == 5
    assert mapping[empty] is empty
    assert mapping[EmptyDerived(a=1)] is empty_derived
    assert mapping[OneSlot(a=1)] is one
    assert mapping[OneSlot(a=1)] is not empty_derived

    assert hash(empty) != hash(empty_derived)
    assert hash(empty_derived) != hash(one)
    assert hash(one) != hash(two)
    assert hash(empty_again) != hash(empty)


def test_immutable_invalid_constructor() -> None:
    with pytest.raises(TypeError):
        Empty(a=1)  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        EmptyDerived(b="two")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        EmptyDerived(a=1, b="two")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        EmptyDerived()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        OneSlot(b="two")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        OneSlot(a=1, b="two")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        OneSlot()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        TwoSlot(b="two")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        TwoSlot(a=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        TwoSlot()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        TwoSlot(a=1, b="two", c="huh?")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        OneSlot(1)  # type: ignore[misc]
    with pytest.raises(TypeError):
        OneSlot(1, 2, 3)  # type: ignore[call-arg, misc]
    with pytest.raises(TypeError):
        OneSlot(1, a=1)  # type: ignore[misc]
