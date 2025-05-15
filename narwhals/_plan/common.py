from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from typing_extensions import Never
    from typing_extensions import TypeAlias

    from narwhals._plan.options import FunctionOptions

    SortOptions: TypeAlias = Any
    SortMultipleOptions: TypeAlias = Any
    WindowType: TypeAlias = Any


class Immutable:
    __slots__ = ()

    def __setattr__(self, name: str, value: Never) -> Never:
        msg = f"{type(self).__name__!r} is immutable, {name!r} cannot be set."
        raise AttributeError(msg)


class ExprIR(Immutable): ...


class Function(ExprIR):
    """Shared by expr functions and namespace functions.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L114
    """

    @property
    def function_options(self) -> FunctionOptions:
        from narwhals._plan.options import FunctionOptions

        return FunctionOptions.default()
