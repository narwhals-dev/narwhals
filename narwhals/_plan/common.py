from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable

    from typing_extensions import Never
    from typing_extensions import Self
    from typing_extensions import TypeAlias
    from typing_extensions import dataclass_transform

    from narwhals._plan.options import FunctionOptions
else:
    # NOTE: This isn't important to the proposal, just wanted IDE support
    # for the **temporary** constructors.
    # It is interesting how much boilerplate this avoids though 🤔
    # https://docs.python.org/3/library/typing.html#typing.dataclass_transform
    def dataclass_transform(
        *,
        eq_default: bool = True,
        order_default: bool = False,
        kw_only_default: bool = False,
        frozen_default: bool = False,
        field_specifiers: tuple[type[Any] | Callable[..., Any], ...] = (),
        **kwargs: Any,
    ) -> Callable[[T], T]:
        def decorator(cls_or_fn: T) -> T:
            cls_or_fn.__dataclass_transform__ = {
                "eq_default": eq_default,
                "order_default": order_default,
                "kw_only_default": kw_only_default,
                "frozen_default": frozen_default,
                "field_specifiers": field_specifiers,
                "kwargs": kwargs,
            }
            return cls_or_fn

        return decorator


T = TypeVar("T")

Seq: TypeAlias = "tuple[T,...]"
"""Immutable Sequence.

Using instead of `Sequence`, as a `list` can be passed there (can't break immutability promise).
"""

Udf: TypeAlias = "Callable[[Any], Any]"
"""Placeholder for `map_batches(function=...)`."""


@dataclass_transform(kw_only_default=True, frozen_default=True)
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

    def __init__(self, **kwds: Any) -> None:
        # NOTE: DUMMY CONSTRUCTOR - don't use beyond prototyping!
        # Just need a quick way to demonstrate `ExprIR` and interactions
        slots: set[str] = set(self.__slots__)
        if not slots and not kwds:
            # NOTE: Fastpath for empty slots
            ...
        elif slots == set(kwds):
            # NOTE: Everything is as expected
            for name, value in kwds.items():
                object.__setattr__(self, name, value)
        elif missing := slots.difference(kwds):
            msg = (
                f"{type(self).__name__!r} requires attributes {sorted(slots)!r}, \n"
                f"but missing values for {sorted(missing)!r}"
            )
            raise TypeError(msg)
        else:
            extra = set(kwds).difference(slots)
            msg = (
                f"{type(self).__name__!r} only supports attributes {sorted(slots)!r}, \n"
                f"but got unknown arguments {sorted(extra)!r}"
            )
            raise TypeError(msg)


class ExprIR(Immutable):
    """Anything that can be a node on a graph of expressions."""

    def to_narwhals(self) -> DummyExpr:
        return DummyExpr._from_ir(self)

    def to_compliant(self) -> DummyCompliantExpr:
        return DummyCompliantExpr._from_ir(self)

    @property
    def is_scalar(self) -> bool:
        return False


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


class DummySeries: ...


class Function(ExprIR):
    """Shared by expr functions and namespace functions.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L114
    """

    @property
    def function_options(self) -> FunctionOptions:
        from narwhals._plan.options import FunctionOptions

        return FunctionOptions.default()

    @property
    def is_scalar(self) -> bool:
        return self.function_options.returns_scalar()
