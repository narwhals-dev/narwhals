from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._plan.common import Immutable, is_expr
from narwhals._plan.dummy import DummyExpr
from narwhals._plan.expr_parsing import (
    parse_into_expr_ir,
    parse_predicates_constraints_into_expr_ir,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from narwhals._plan.common import ExprIR
    from narwhals._plan.expr import Ternary
    from narwhals._plan.typing import IntoExpr, IntoExprColumn, Seq


class When(Immutable):
    __slots__ = ("condition",)
    condition: ExprIR

    def then(self, expr: IntoExpr, /) -> Then:
        return Then(condition=self.condition, statement=parse_into_expr_ir(expr))

    @staticmethod
    def _from_expr(expr: DummyExpr, /) -> When:
        return When(condition=expr._ir)

    @staticmethod
    def _from_ir(ir: ExprIR, /) -> When:
        return When(condition=ir)


class Then(Immutable, DummyExpr):
    __slots__ = ("condition", "statement")
    condition: ExprIR
    statement: ExprIR

    def when(
        self, *predicates: IntoExprColumn | Iterable[IntoExprColumn], **constraints: Any
    ) -> ChainedWhen:
        condition = parse_predicates_constraints_into_expr_ir(*predicates, **constraints)
        return ChainedWhen(
            conditions=(self.condition, condition), statements=(self.statement,)
        )

    def otherwise(self, statement: IntoExpr, /) -> DummyExpr:
        return self._from_ir(self._otherwise(statement))

    def _otherwise(self, statement: IntoExpr = None, /) -> ExprIR:
        return ternary_expr(self.condition, self.statement, parse_into_expr_ir(statement))

    @property
    def _ir(self) -> ExprIR:  # type: ignore[override]
        return self._otherwise()

    @classmethod
    def _from_ir(cls, ir: ExprIR, /) -> DummyExpr:  # type: ignore[override]
        return DummyExpr._from_ir(ir)

    def __eq__(self, value: object) -> DummyExpr | bool:  # type: ignore[override]
        if is_expr(value):
            return super(DummyExpr, self).__eq__(value)
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


class ChainedThen(Immutable, DummyExpr):
    """https://github.com/pola-rs/polars/blob/b9dd8cdbd6e6ec8373110536955ed5940b9460ec/crates/polars-plan/src/dsl/arity.rs#L89-L130."""

    __slots__ = ("conditions", "statements")
    conditions: Seq[ExprIR]
    statements: Seq[ExprIR]

    def when(
        self, *predicates: IntoExprColumn | Iterable[IntoExprColumn], **constraints: Any
    ) -> ChainedWhen:
        condition = parse_predicates_constraints_into_expr_ir(*predicates, **constraints)
        return ChainedWhen(
            conditions=(*self.conditions, condition), statements=self.statements
        )

    def otherwise(self, statement: IntoExpr, /) -> DummyExpr:
        return self._from_ir(self._otherwise(statement))

    def _otherwise(self, statement: IntoExpr = None, /) -> ExprIR:
        otherwise = parse_into_expr_ir(statement)
        it_conditions = reversed(self.conditions)
        it_statements = reversed(self.statements)
        for e in it_conditions:
            otherwise = ternary_expr(e, next(it_statements), otherwise)
        return otherwise

    @property
    def _ir(self) -> ExprIR:  # type: ignore[override]
        return self._otherwise()

    @classmethod
    def _from_ir(cls, ir: ExprIR, /) -> DummyExpr:  # type: ignore[override]
        return DummyExpr._from_ir(ir)

    def __eq__(self, value: object) -> DummyExpr | bool:  # type: ignore[override]
        if is_expr(value):
            return super(DummyExpr, self).__eq__(value)
        return super().__eq__(value)


def ternary_expr(predicate: ExprIR, truthy: ExprIR, falsy: ExprIR, /) -> Ternary:
    from narwhals._plan.expr import Ternary

    return Ternary(predicate=predicate, truthy=truthy, falsy=falsy)
