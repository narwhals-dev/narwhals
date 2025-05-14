from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Never


class Immutable:
    __slots__ = ()

    def __setattr__(self, name: str, value: Never) -> Never:
        msg = f"{type(self).__name__!r} is immutable, {name!r} cannot be set."
        raise AttributeError(msg)


class ExprIR(Immutable): ...


class Function(ExprIR):
    """Shared by expr functions and namespace functions."""


class FunctionExpr(ExprIR): ...
