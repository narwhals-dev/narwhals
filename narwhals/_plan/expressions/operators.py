from __future__ import annotations

import operator as op
from typing import TYPE_CHECKING

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._guards import is_function_expr
from narwhals._plan._immutable import Immutable
from narwhals._plan.exceptions import (
    binary_expr_length_changing_error,
    binary_expr_shape_error,
)

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from typing_extensions import Self, TypeAlias

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

    BinaryAny: TypeAlias = BinaryExpr[Any, Any, Any]


class Operator(Immutable):
    _func: ClassVar[OperatorFn]
    _symbol: ClassVar[str]
    __expr_ir_dtype__: ClassVar[ResolveDType] = ResolveDType()

    def __repr__(self) -> str:
        return self._symbol

    def __init_subclass__(
        cls,
        *args: Any,
        func: OperatorFn | None,
        symbol: str = "",
        dtype: DType | None = None,
        **kwds: Any,
    ) -> None:
        super().__init_subclass__(*args, **kwds)
        if func:
            cls._func = func
        cls._symbol = symbol or cls.__name__
        if dtype is not None:
            cls.__expr_ir_dtype__ = ResolveDType.just_dtype(dtype)

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

    def resolve_dtype(self, node: BinaryAny, schema: FrozenSchema, /) -> DType:
        return self.__expr_ir_dtype__(node, schema)


def _is_filtration(ir: ExprIR) -> bool:
    return not ir.is_scalar and is_function_expr(ir) and not ir.options.is_elementwise()


class SelectorOperator(Operator, func=None):
    """Operators that can *also* be used in selectors."""

    def to_binary_selector(
        self, left: LeftSelectorT, right: RightSelectorT, /
    ) -> BinarySelector[LeftSelectorT, Self, RightSelectorT]:
        from narwhals._plan.expressions.expr import BinarySelector

        return BinarySelector(left=left, op=self, right=right)


# fmt: off
class Eq(Operator, func=op.eq, symbol="==", dtype=dtm.BOOL): ...
class NotEq(Operator, func=op.ne, symbol="!=", dtype=dtm.BOOL): ...
class Lt(Operator, func=op.le, symbol="<", dtype=dtm.BOOL): ...
class LtEq(Operator, func=op.lt, symbol="<=", dtype=dtm.BOOL): ...
class Gt(Operator, func=op.gt, symbol=">", dtype=dtm.BOOL): ...
class GtEq(Operator, func=op.ge, symbol=">=", dtype=dtm.BOOL): ...
class And(SelectorOperator, func=op.and_, symbol="&", dtype=dtm.BOOL): ...
class Or(SelectorOperator, func=op.or_, symbol="|", dtype=dtm.BOOL): ...
class ExclusiveOr(SelectorOperator, func=op.xor, symbol="^", dtype=dtm.BOOL): ...
# TODO @dangotbanned: Review adding a subset of `get_arithmetic_field` *after* `get_supertype`
# https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/schema.rs#L475-L766
class Arithmetic(Operator, func=None): ...
class TrueDivide(Arithmetic, func=op.truediv, symbol="/"):
    def resolve_dtype(self, node: BinaryAny, schema: FrozenSchema, /) -> DType:
        left, right = node.left.resolve_dtype(schema), node.right.resolve_dtype(schema)
        return dtm.truediv_dtype(left, right)
class Add(Arithmetic, func=op.add, symbol="+"): ...
class Sub(SelectorOperator, func=op.sub, symbol="-"): ...
class Multiply(Arithmetic, func=op.mul, symbol="*"): ...
class FloorDivide(Arithmetic, func=op.floordiv, symbol="//"): ...
class Modulus(Arithmetic, func=op.mod, symbol="%"): ...
# fmt: on
