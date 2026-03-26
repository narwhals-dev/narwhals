from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._parse import parse_into_expr_ir
from narwhals._plan.expressions.namespace import ExprNamespace
from narwhals._plan.expressions.strings import IRStringNamespace

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr


class ExprStringNamespace(ExprNamespace[IRStringNamespace]):
    @property
    def _ir_namespace(self) -> type[IRStringNamespace]:
        return IRStringNamespace

    def len_chars(self) -> Expr:
        return self._with_unary(self._ir.len_chars())

    def replace(
        self, pattern: str, value: str | Expr, *, literal: bool = False, n: int = 1
    ) -> Expr:
        other = parse_into_expr_ir(value, str_as_lit=True)
        replace = self._ir.replace(pattern, literal=literal, n=n)
        return self._expr._from_ir(replace.to_function_expr(self._expr._ir, other))

    def replace_all(
        self, pattern: str, value: str | Expr, *, literal: bool = False
    ) -> Expr:
        other = parse_into_expr_ir(value, str_as_lit=True)
        replace = self._ir.replace_all(pattern, literal=literal)
        return self._expr._from_ir(replace.to_function_expr(self._expr._ir, other))

    def strip_chars(self, characters: str | None = None) -> Expr:
        return self._with_unary(self._ir.strip_chars(characters))

    def starts_with(self, prefix: str) -> Expr:
        return self._with_unary(self._ir.starts_with(prefix=prefix))

    def ends_with(self, suffix: str) -> Expr:
        return self._with_unary(self._ir.ends_with(suffix=suffix))

    def contains(self, pattern: str, *, literal: bool = False) -> Expr:
        return self._with_unary(self._ir.contains(pattern, literal=literal))

    def slice(self, offset: int, length: int | None = None) -> Expr:
        return self._with_unary(self._ir.slice(offset, length))

    def head(self, n: int = 5) -> Expr:
        return self._with_unary(self._ir.head(n))

    def tail(self, n: int = 5) -> Expr:
        return self._with_unary(self._ir.tail(n))

    def split(self, by: str) -> Expr:
        return self._with_unary(self._ir.split(by=by))

    def to_date(self, format: str | None = None) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.to_date(format))

    def to_datetime(self, format: str | None = None) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.to_datetime(format))

    def to_lowercase(self) -> Expr:
        return self._with_unary(self._ir.to_lowercase())

    def to_uppercase(self) -> Expr:
        return self._with_unary(self._ir.to_uppercase())

    def to_titlecase(self) -> Expr:
        return self._with_unary(self._ir.to_titlecase())

    def zfill(self, length: int) -> Expr:
        return self._with_unary(self._ir.zfill(length=length))
