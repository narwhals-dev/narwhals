from __future__ import annotations

import re
import string
from itertools import repeat
from typing import TYPE_CHECKING, Any, TypeVar

import pytest

from narwhals._plan import when_then
from narwhals._plan._immutable import Immutable

if TYPE_CHECKING:
    from collections.abc import Iterator
T_co = TypeVar("T_co", covariant=True)


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


def test_immutable_hash_cache() -> None:
    int_long = 9999999999999999999999999999999999999999999999999999999999
    str_long = "\n".join(repeat(string.printable, 100))
    obj = TwoSlot(a=int_long, b=str_long)

    with pytest.raises(AttributeError):
        _ = getattr(obj, "__immutable_hash_value__")  # noqa: B009

    hash_cache_miss = hash(obj)
    cached = getattr(obj, "__immutable_hash_value__")  # noqa: B009
    hash_cache_hit = hash(obj)
    assert hash_cache_miss == cached == hash_cache_hit


def _collect_immutable_descendants() -> list[type[Immutable]]:
    # NOTE: Will populate `__subclasses__` by bringing the defs into scope
    from narwhals._plan import (
        _expansion,
        _expr_ir,
        _function,
        expressions,
        options,
        schema,
        when_then,
    )

    _ = expressions, schema, options, _expansion, _expr_ir, _function, when_then
    return sorted(set(_iter_descendants(Immutable)), key=repr)


def _iter_descendants(*bases: type[T_co]) -> Iterator[type[T_co]]:
    seen = set[T_co]()
    for base in bases:
        yield base
        if (children := (base.__subclasses__())) and (
            unseen := set(children).difference(seen)
        ):
            yield from _iter_descendants(*unseen)


ALLOW_DICT_TO_AVOID_MULTIPLE_BASES_HAVE_INSTANCE_LAYOUT_CONFLICT_ERROR = frozenset(
    (when_then.Then, when_then.ChainedThen)
)


@pytest.fixture(params=_collect_immutable_descendants(), ids=lambda tp: tp.__name__)
def immutable_type(request: pytest.FixtureRequest) -> type[Immutable]:
    tp: type[Immutable] = request.param
    request.applymarker(
        pytest.mark.xfail(
            tp in ALLOW_DICT_TO_AVOID_MULTIPLE_BASES_HAVE_INSTANCE_LAYOUT_CONFLICT_ERROR,
            reason="Multiple inheritance + `__slots__` = bad",
        )
    )
    return tp


def test_immutable___slots___(immutable_type: type[Immutable]) -> None:
    featureless_instance = object.__new__(immutable_type)

    # NOTE: If this fails, `__setattr__` has been overridden
    with pytest.raises(AttributeError, match=r"immutable"):
        featureless_instance.i_dont_exist = 999  # type: ignore[assignment]

    # NOTE: If this fails, `__slots__` lose the size benefit
    with pytest.raises(AttributeError, match=re.escape("has no attribute '__dict__'")):
        _ = featureless_instance.__dict__

    slots = immutable_type.__slots__
    if slots:
        assert len(slots) != 0, slots
