from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from typing_extensions import Never
    from typing_extensions import Self

    from narwhals._plan.options import FunctionOptions


class Immutable:
    __slots__ = ()

    def __setattr__(self, name: str, value: Never) -> Never:
        msg = f"{type(self).__name__!r} is immutable, {name!r} cannot be set."
        raise AttributeError(msg)

    def __init_subclass__(cls, *args: Any, **kwds: Any) -> None:
        super().__init_subclass__(*args, **kwds)
        if cls.__slots__:
            ...
        else:
            cls.__slots__ = ()

    def __hash__(self) -> int:
        empty = object()
        slots: tuple[str, ...] = self.__slots__
        return hash(tuple(getattr(self, name, empty) for name in slots))

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        elif type(self) is not type(other):
            return False
        empty = object()
        slots: tuple[str, ...] = self.__slots__
        return all(
            getattr(self, name, empty) == getattr(other, name, empty) for name in slots
        )


class ExprIR(Immutable):
    """Anything that can be a node on a graph of expressions."""

    def to_narwhals(self) -> DummyExpr:
        return DummyExpr._from_ir(self)

    def to_compliant(self) -> DummyCompliantExpr:
        return DummyCompliantExpr._from_ir(self)


# NOTE: Overly simplified placeholders for mocking typing
# Entirely ignoring namespace + function binding
class DummyExpr:
    _ir: ExprIR

    @classmethod
    def _from_ir(cls, ir: ExprIR, /) -> Self:
        obj = cls.__new__(cls)
        obj._ir = ir
        return obj


class DummyCompliantExpr:
    _ir: ExprIR

    @classmethod
    def _from_ir(cls, ir: ExprIR, /) -> Self:
        obj = cls.__new__(cls)
        obj._ir = ir
        return obj


class Function(ExprIR):
    """Shared by expr functions and namespace functions.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L114
    """

    @property
    def function_options(self) -> FunctionOptions:
        from narwhals._plan.options import FunctionOptions

        return FunctionOptions.default()
