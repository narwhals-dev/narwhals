from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._plan._guards import is_expr
from narwhals._plan._immutable import Immutable
from narwhals._plan._parse import (
    parse_into_expr_ir as _parse_into_expr_ir,
    parse_predicates_constraints_into_expr_ir,
)
from narwhals._plan.expr import Expr
from narwhals.exceptions import MultiOutputExpressionError

if TYPE_CHECKING:
    from narwhals._plan.expressions import ExprIR, TernaryExpr
    from narwhals._plan.typing import IntoExpr, IntoExprColumn, OneOrIterable, Seq


def _multi_output_error(expr: ExprIR) -> MultiOutputExpressionError:
    msg = f"Multi-output expressions are not supported in a `when-then-otherwise` context.\n{expr!r}"
    return MultiOutputExpressionError(msg)


def parse_into_expr_ir(statement: IntoExpr, /) -> ExprIR:
    expr_ir = _parse_into_expr_ir(statement)
    if expr_ir.meta.has_multiple_outputs():
        raise _multi_output_error(expr_ir)
    return expr_ir


class When(Immutable):
    __slots__ = ("condition",)
    condition: ExprIR

    def then(self, expr: IntoExpr, /) -> Then:
        return Then(condition=self.condition, statement=parse_into_expr_ir(expr))

    @staticmethod
    def _from_expr(expr: Expr, /) -> When:
        return When(condition=expr._ir)

    @staticmethod
    def _from_ir(expr_ir: ExprIR, /) -> When:
        return When(condition=expr_ir)


class Then(Immutable, Expr):
    __slots__ = ("condition", "statement")
    condition: ExprIR
    statement: ExprIR

    def when(
        self, *predicates: OneOrIterable[IntoExprColumn], **constraints: Any
    ) -> ChainedWhen:
        condition = parse_predicates_constraints_into_expr_ir(*predicates, **constraints)
        return ChainedWhen(
            conditions=(self.condition, condition), statements=(self.statement,)
        )

    def otherwise(self, statement: IntoExpr, /) -> Expr:
        return self._from_ir(self._otherwise(statement))

    def _otherwise(self, statement: IntoExpr = None, /) -> ExprIR:
        return ternary_expr(self.condition, self.statement, parse_into_expr_ir(statement))

    @property
    def _ir(self) -> ExprIR:  # type: ignore[override]
        return self._otherwise()

    @classmethod
    def _from_ir(cls, expr_ir: ExprIR, /) -> Expr:  # type: ignore[override]
        return Expr._from_ir(expr_ir)

    def __eq__(self, value: object) -> Expr | bool:  # type: ignore[override]
        if is_expr(value):
            return super(Expr, self).__eq__(value)
        return super().__eq__(value)


class ChainedWhen(Immutable):
    __slots__ = ("conditions", "statements")
    conditions: Seq[ExprIR]
    statements: Seq[ExprIR]

    def then(self, statement: IntoExpr, /) -> ChainedThen:
        return ChainedThen(
            conditions=self.conditions,
            statements=(*self.statements, parse_into_expr_ir(statement)),
        )


class ChainedThen(Immutable, Expr):
    __slots__ = ("conditions", "statements")
    conditions: Seq[ExprIR]
    statements: Seq[ExprIR]

    def when(
        self, *predicates: OneOrIterable[IntoExprColumn], **constraints: Any
    ) -> ChainedWhen:
        condition = parse_predicates_constraints_into_expr_ir(*predicates, **constraints)
        return ChainedWhen(
            conditions=(*self.conditions, condition), statements=self.statements
        )

    def otherwise(self, statement: IntoExpr, /) -> Expr:
        return self._from_ir(self._otherwise(statement))

    def _otherwise(self, statement: IntoExpr = None, /) -> ExprIR:
        otherwise = parse_into_expr_ir(statement)
        for cond, stmt in zip(reversed(self.conditions), reversed(self.statements)):
            otherwise = ternary_expr(cond, stmt, otherwise)
        return otherwise

    @property
    def _ir(self) -> ExprIR:  # type: ignore[override]
        return self._otherwise()

    @classmethod
    def _from_ir(cls, expr_ir: ExprIR, /) -> Expr:  # type: ignore[override]
        return Expr._from_ir(expr_ir)

    def __eq__(self, value: object) -> Expr | bool:  # type: ignore[override]
        if is_expr(value):
            return super(Expr, self).__eq__(value)
        return super().__eq__(value)


def ternary_expr(predicate: ExprIR, truthy: ExprIR, falsy: ExprIR, /) -> TernaryExpr:
    from narwhals._plan.expressions.expr import TernaryExpr

    return TernaryExpr(predicate=predicate, truthy=truthy, falsy=falsy)
