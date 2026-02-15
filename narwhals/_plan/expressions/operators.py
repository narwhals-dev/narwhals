from __future__ import annotations

import operator as op
from typing import TYPE_CHECKING

from narwhals._plan._guards import is_function_expr
from narwhals._plan._immutable import Immutable
from narwhals._plan.dtypes_mapper import BOOLEAN_DTYPE
from narwhals._plan.exceptions import (
    binary_expr_length_changing_error,
    binary_expr_shape_error,
)

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from typing_extensions import Self

    from narwhals._plan.expressions import BinaryExpr, BinarySelector, ExprIR
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import (
        LeftSelectorT,
        LeftT,
        OperatorFn,
        RightSelectorT,
        RightT,
    )
    from narwhals.dtypes import DType


class Operator(Immutable):
    _func: ClassVar[OperatorFn]
    _symbol: ClassVar[str]

    def __repr__(self) -> str:
        return self._symbol

    def __init_subclass__(
        cls, *args: Any, func: OperatorFn | None, symbol: str = "", **kwds: Any
    ) -> None:
        super().__init_subclass__(*args, **kwds)
        if func:
            cls._func = func
        cls._symbol = symbol or cls.__name__

    def to_binary_expr(
        self, left: LeftT, right: RightT, /
    ) -> BinaryExpr[LeftT, Self, RightT]:
        from narwhals._plan.expressions.expr import BinaryExpr

        if _is_filtration(left):
            if _is_filtration(right):
                raise binary_expr_length_changing_error(left, self, right)
            if not right.is_scalar:
                raise binary_expr_shape_error(left, self, right)
        elif _is_filtration(right):
            if not left.is_scalar:
                raise binary_expr_shape_error(left, self, right)
        return BinaryExpr(left=left, op=self, right=right)

    def __call__(self, lhs: Any, rhs: Any) -> Any:
        """Apply binary operator to `left`, `right` operands."""
        return self.__class__._func(lhs, rhs)

    def _resolve_dtype(self, schema: FrozenSchema, left: ExprIR, right: ExprIR) -> DType:
        msg = f"`NamedIR[...].resolve_dtype()` is not yet implemented for {self!r}\n"
        f"[({left!r}) {self!r} ({right!r})]"
        raise NotImplementedError(msg)


def _is_filtration(ir: ExprIR) -> bool:
    return not ir.is_scalar and is_function_expr(ir) and not ir.options.is_elementwise()


class SelectorOperator(Operator, func=None):
    """Operators that can *also* be used in selectors."""

    def to_binary_selector(
        self, left: LeftSelectorT, right: RightSelectorT, /
    ) -> BinarySelector[LeftSelectorT, Self, RightSelectorT]:
        from narwhals._plan.expressions.expr import BinarySelector

        return BinarySelector(left=left, op=self, right=right)


class Logical(Operator, func=None):
    def _resolve_dtype(self, schema: FrozenSchema, left: ExprIR, right: ExprIR) -> DType:
        return BOOLEAN_DTYPE


# TODO @dangotbanned: Review adding a subset of `get_arithmetic_field` *after* `get_supertype`
class Arithmetic(Operator, func=None):
    # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/schema.rs#L475-L766
    # NOTE: Deferred due to complexity that is outside of `get_supertype`
    # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/schema.rs#L741-L877
    ...


# TODO @dangotbanned: Review if needed for mro ambiguity
class SelectorLogical(SelectorOperator, func=None):
    def _resolve_dtype(self, schema: FrozenSchema, left: ExprIR, right: ExprIR) -> DType:
        return BOOLEAN_DTYPE


class SelectorArithmetic(SelectorOperator, func=None): ...


# fmt: off
class Eq(Logical, func=op.eq, symbol="=="): ...
class NotEq(Logical, func=op.ne, symbol="!="): ...
class Lt(Logical, func=op.le, symbol="<"): ...
class LtEq(Logical, func=op.lt, symbol="<="): ...
class Gt(Logical, func=op.gt, symbol=">"): ...
class GtEq(Logical, func=op.ge, symbol=">="): ...
class Add(Arithmetic, func=op.add, symbol="+"): ...
class Sub(SelectorArithmetic, func=op.sub, symbol="-"): ...
class Multiply(Arithmetic, func=op.mul, symbol="*"): ...
class TrueDivide(Arithmetic, func=op.truediv, symbol="/"): ...
class FloorDivide(Arithmetic, func=op.floordiv, symbol="//"): ...
class Modulus(Arithmetic, func=op.mod, symbol="%"): ...
class And(SelectorLogical, func=op.and_, symbol="&"): ...
class Or(SelectorLogical, func=op.or_, symbol="|"): ...
class ExclusiveOr(SelectorLogical, func=op.xor, symbol="^"): ...
# fmt: on
