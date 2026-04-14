from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, ClassVar, Final, Generic, Literal, final

from narwhals._plan._meta import SlottedMeta
from narwhals._plan.exceptions import (
    function_arg_non_scalar_error,
    function_arity_error,
    function_expr_invalid_operation_error,
)
from narwhals._plan.typing import Seq
from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import TypeAlias

    from narwhals._plan.expressions import ExprIR, Function

__all__ = ["SCALAR", "Binary", "Parameters", "Ternary", "Unary", "Variadic"]

_ArgsT_co = TypeVar(
    "_ArgsT_co", bound="Seq[ExprIR]", default="Seq[ExprIR]", covariant=True
)
_Arity: TypeAlias = Literal[1, 2, 3, "*"]


class Constraint(enum.Enum):
    """A rule for an expression argument to a function."""

    ANY = enum.auto()
    """Place no restrictions on the argument."""
    DEFAULT = enum.auto()
    """Reject scalars when used in a non-elementwise function."""
    SCALAR = enum.auto()
    """Require an argument to be scalar."""


_ANY: Final = Constraint.ANY
_DEFAULT: Final = Constraint.DEFAULT


# TODO @dangotbanned: Docs
# - Unary by default
# - Each parameter can have it's own rule
# TODO @dangotbanned: Feat
# - Reprs
# - Port dispatch helpers from `arrow.{namespace,expr}` into something that uses this
class Parameters(Generic[_ArgsT_co], metaclass=SlottedMeta):
    __slots__ = ("_constraints",)
    _constraints: tuple[Constraint, ...]
    _arity: ClassVar[_Arity]

    def check(self, function: Function, exprs: Seq[ExprIR], /) -> Seq[ExprIR]:
        for constraint, expr in self._zip_constraints(function, exprs):
            if constraint is _ANY:
                continue
            if constraint is _DEFAULT:
                # NOTE: SQL-like requires elementwise here, otherwise `is_length_preserving` would be sufficient
                if expr.is_scalar() and not function.is_elementwise():
                    raise function_expr_invalid_operation_error(function, expr)
                continue
            if constraint is SCALAR and not expr.is_scalar():
                raise function_arg_non_scalar_error(function, expr)
        return exprs

    def unwrap(self, exprs: Seq[ExprIR]) -> _ArgsT_co:
        """Ensure we have the correct number of expressions.

        This shouldn't be used before expression expansion.
        """
        raise NotImplementedError

    def _zip_constraints(
        self, function: Function, exprs: Seq[ExprIR], /
    ) -> Iterator[tuple[Constraint, ExprIR]]:
        # `zip(strict=True)` produces a very unhelpful error message
        if constraints := self._constraints:
            if len(exprs) != len(constraints):
                raise function_arity_error(function, self._arity, exprs)
            yield from zip(constraints, exprs)

    def __init_subclass__(cls, *, arity: _Arity, **_: Any) -> None:
        super().__init_subclass__(**_)
        cls._arity = arity


@final
class Unary(Parameters[tuple["ExprIR"]], arity=1):
    def __init__(self, arg: Constraint = _DEFAULT, /) -> None:
        self._constraints: tuple[Constraint] = (arg,)

    def unwrap(self, exprs: Seq[ExprIR]) -> tuple[ExprIR]:  # pragma: no cover
        (a,) = exprs
        return (a,)


@final
class Binary(Parameters[tuple["ExprIR", "ExprIR"]], arity=2):
    def __init__(self, left: Constraint = _DEFAULT, right: Constraint = _ANY) -> None:
        self._constraints: tuple[Constraint, Constraint] = left, right

    def unwrap(self, exprs: Seq[ExprIR]) -> tuple[ExprIR, ExprIR]:  # pragma: no cover
        a, b = exprs
        return (a, b)


@final
class Ternary(Parameters[tuple["ExprIR", "ExprIR", "ExprIR"]], arity=3):
    def __init__(
        self,
        arg_1: Constraint = _DEFAULT,
        arg_2: Constraint = _ANY,
        arg_3: Constraint = _ANY,
    ) -> None:
        self._constraints: tuple[Constraint, Constraint, Constraint] = arg_1, arg_2, arg_3

    def unwrap(
        self, exprs: Seq[ExprIR]
    ) -> tuple[ExprIR, ExprIR, ExprIR]:  # pragma: no cover
        a, b, c = exprs
        return (a, b, c)


# TODO @dangotbanned: Restrict variadic to elementwise + scalar?
@final
class Variadic(Parameters[Seq["ExprIR"]], arity="*"):
    def __init__(self) -> None:
        self._constraints = ()

    def unwrap(self, exprs: Seq[ExprIR]) -> Seq[ExprIR]:  # pragma: no cover
        return exprs


SCALAR: Final = Constraint.SCALAR
"""Require an argument to be scalar."""
