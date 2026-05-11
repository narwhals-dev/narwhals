from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from narwhals._plan._nodes import node, nodes
from narwhals._plan.expressions import ExprIR
from tests.plan.utils import re_compile


def test_known_field_specifiers() -> None:
    """Mainly focused on static typing being understood.

    Runtime errors aren't important.
    """

    class E1(ExprIR): ...

    class E2(ExprIR):
        __slots__ = ("a", "b", "c")
        a: int
        b: ExprIR = node()
        c: str

    class E3(ExprIR):
        __slots__ = ("a", "b", "c")
        c: tuple[ExprIR, ...] = nodes()
        a: ExprIR = node(observe_scalar=False)
        b: int

    class E4(ExprIR):
        __slots__ = ("a", "b", "c")
        b: tuple[ExprIR, ...] = nodes()
        c: ExprIR = node()
        a: ExprIR = node(observe_scalar=False)

    class E1S1(E1): ...

    class E1S2(E1):
        __slots__ = ("a",)
        a: ExprIR = node()

    class E2S1(E2):
        __slots__ = ("d",)
        d: float

    class E2S2(E2):
        __slots__ = ("d",)
        d: tuple[ExprIR, ...] = nodes()

    e1 = E1()
    e2 = E2(a=1, b=e1, c="c")
    e3 = E3(a=e2, b=1, c=(e1, e2))
    e4 = E4(a=e1, b=(e3, e1), c=e2)

    e1s1 = E1S1()
    e1s2 = E1S2(a=e4)
    e2s1 = E2S1(a=1, b=e1, c="c", d=1.0)
    e2s2 = E2S2(a=1, b=e1, c="c", d=(e2s1,))

    with pytest.raises(TypeError):
        E1(a=1, b=e1, c="c", d=(e2s1,))  # pyright: ignore[reportCallIssue]

    with pytest.raises(TypeError):
        E2(a=1, b=e1, c="c", d=1.0)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        E2(a=1, c="c")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        E3(a=e4)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        E4()  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        E1S1(e2s2)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        E1S2(a=e1, b=(e3, e1), c=e2)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        E2S1(a=e1s2, b=1, c=(e1, e1s1))  # type: ignore[call-arg, arg-type]

    with pytest.raises(TypeError):
        E2S2(a=1, b=e1, c="c")  # type: ignore[call-arg]

    E4(a=e1, b=e4, c=(e3, e1))  # type: ignore[arg-type]
    E2S2(a=1, b=e1, c="c", d=1.0)  # type: ignore[arg-type]


def test_unknown_field_specifiers() -> None:
    """Entire focus is runtime errors.

    The traceback (on `3.13`) points to the `class` statement:

        class NotGood(ExprIR): # <

    But the issue is after that point:

        class NotGood(ExprIR):
            __slots__ = ("bad",)
        #                 ^^^
            bad: tuple[str, ...] = ...
        #   ^^^                    ^^^

    So this error should help you get back on track
    """
    pattern = re_compile(
        r"ExprIR subclass.+tried to assign a .+ to 'bad'.+ also.+__slots__.+"
        r"This syntax is reserved for field specifiers.+node\(\).+nodes\(\)"
    )
    with pytest.raises(TypeError, match=pattern):

        class NotGood(ExprIR):
            __slots__ = ("bad",)
            bad: tuple[str, ...] = dataclasses.field(default_factory=tuple)

    with pytest.raises(TypeError, match=pattern):

        class AlsoBad(ExprIR):
            __slots__ = ("bad",)
            bad = dataclasses.field(default=1)

    with pytest.raises(TypeError, match=pattern):

        class StillBanned(ExprIR):
            __slots__ = ("bad",)
            bad: int = 1

    def fraud() -> Any:
        return "nope!"

    with pytest.raises(TypeError, match=pattern):

        class AndThisToo(ExprIR):
            __slots__ = ("bad",)
            bad: str = fraud()
