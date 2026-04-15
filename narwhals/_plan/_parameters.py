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
Arity: TypeAlias = Literal[1, 2, 3, "*"]
"""The number of expression arguments taken by a function.

## See Also
[Arity](https://en.wikipedia.org/wiki/Arity)
"""


class Constraint(enum.Enum):
    """A rule for an expression argument to a function."""

    ANY = "Place no restrictions on the argument"
    DEFAULT = "Reject scalars in non-elementwise functions"
    SCALAR = "Require an argument to be scalar"


SCALAR: Final = Constraint.SCALAR
"""Require an argument to be scalar."""
_ANY: Final = Constraint.ANY
_DEFAULT: Final = Constraint.DEFAULT


# TODO @dangotbanned: Feat
# - Port dispatch helpers from `arrow.{namespace,expr}` into something that uses this
class Parameters(Generic[_ArgsT_co], metaclass=SlottedMeta):
    """Expectations of expression arguments to a function.

    Default instances encode how many:
    >>> Ternary().arity
    3

    And define rules each must follow:
    >>> Binary()
    Binary(DEFAULT, ANY)

    We can constrain that further if needed:
    >>> Binary(right=Constraint.SCALAR)
    Binary(DEFAULT, SCALAR)

    Putting it all together:
    >>> import narwhals._plan as nw
    >>> from narwhals._plan import expressions as ir

    Both inputs must be scalar:
    >>> ir.ranges.IntRange.__function_parameters__
    Binary(SCALAR, SCALAR)

    Which permits literals and aggregations:
    >>> nw.int_range(0, nw.len())._ir
    int_range([lit(int: 0), len()])

    But will raise on anything else:
    >>> nw.int_range(0, nw.col("bad").abs())  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ShapeError: `int_range()` does not support non-scalar expressions, got: `col('bad').abs()`.

    ## See Also
    [Argument vs parameter](https://docs.python.org/3/faq/programming.html#faq-argument-vs-parameter)
    """

    __slots__ = ("_constraints",)
    _constraints: tuple[Constraint, ...]
    _arity: ClassVar[Arity]

    @property
    def arity(self) -> Arity:
        """The number of expression arguments taken by a function.

        >>> [tp().arity for tp in (Unary, Binary, Ternary, Variadic)]
        [1, 2, 3, '*']
        """
        return self._arity

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
                raise function_arity_error(function, self.arity, exprs)
            yield from zip(constraints, exprs)

    def __init_subclass__(cls, *, arity: Arity, **_: Any) -> None:
        super().__init_subclass__(**_)
        cls._arity = arity

    def __repr__(self) -> str:
        params = (
            ", ".join(c.name for c in cons) if (cons := self._constraints) else self.arity
        )
        return f"{type(self).__name__}({params})"


@final
class Unary(Parameters[tuple["ExprIR"]], arity=1):
    """Takes one expression argument.

    This is the default and used for most functions (by count).

    The expression is whatever we had when the `Expr` *method* was called:
    >>> import narwhals._plan as nw
    >>> expr = nw.col("a").abs()
    >>> expr._ir.parameters
    Unary(DEFAULT)
    >>> expr._ir.input[0]
    col('a')
    """

    def __init__(self, arg: Constraint = _DEFAULT, /) -> None:
        self._constraints: tuple[Constraint] = (arg,)

    def unwrap(self, exprs: Seq[ExprIR]) -> tuple[ExprIR]:  # pragma: no cover
        (a,) = exprs
        return (a,)


@final
class Binary(Parameters[tuple["ExprIR", "ExprIR"]], arity=2):
    """Takes two expression arguments.

    In many cases, this is like `Unary` but the `Expr` *method* accepts an expression:
    >>> import narwhals._plan as nw
    >>> expr = nw.col("a").fill_null(nw.col("a").min())
    >>> print(f"{expr._ir.parameters} | {expr._ir.input}")
    Binary(DEFAULT, ANY) | (col('a'), col('a').min())

    We also use this to represent range *functions*:
    >>> expr = nw.int_range(0, 10)
    >>> print(f"{expr._ir.parameters} | {expr._ir.input}")
    Binary(SCALAR, SCALAR) | (lit(int: 0), lit(int: 10))
    """

    def __init__(self, left: Constraint = _DEFAULT, right: Constraint = _ANY) -> None:
        self._constraints: tuple[Constraint, Constraint] = left, right

    def unwrap(self, exprs: Seq[ExprIR]) -> tuple[ExprIR, ExprIR]:  # pragma: no cover
        a, b = exprs
        return (a, b)


@final
class Ternary(Parameters[tuple["ExprIR", "ExprIR", "ExprIR"]], arity=3):
    """Takes three expression arguments.

    This is like `Unary` but the `Expr` *method* accepts two expressions:
    >>> import narwhals._plan as nw
    >>> expr = nw.col("a").alias("clip").clip(nw.col("b"), nw.col("c"))
    >>> print(f"{expr._ir.parameters} | {expr._ir.input}")
    Ternary(DEFAULT, ANY, ANY) | (col('a').alias('clip'), col('b'), col('c'))
    """

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
    """Takes a variable number of expression arguments.

    Describes the parameters of horizontal functions:
    >>> import narwhals._plan as nw
    >>> expr = nw.all_horizontal("a", "b", "c", "d")
    >>> print(f"{expr._ir.parameters} | {expr._ir.input}")
    Variadic(*) | (col('a'), col('b'), col('c'), col('d'))

    Yep, this too:
    >>> expr = nw.concat_str(nw.col("c"), nw.nth(-1))
    >>> print(f"{expr._ir.parameters} | {expr._ir.input}")
    Variadic(*) | (col('c'), ncs.last())
    """

    def __init__(self) -> None:
        self._constraints = ()

    def unwrap(self, exprs: Seq[ExprIR]) -> Seq[ExprIR]:  # pragma: no cover
        return exprs
