from __future__ import annotations

from typing import TYPE_CHECKING

import daft

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import StringNamespace

if TYPE_CHECKING:
    from daft import Expression

    from narwhals._daft.expr import DaftExpr


class DaftExprStringNamespace(LazyExprNamespace["DaftExpr"], StringNamespace["DaftExpr"]):
    def __init__(self, expr: DaftExpr) -> None:
        self.compliant = expr

    def starts_with(self, prefix: str) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.str.startswith(prefix))

    def ends_with(self, prefix: str) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.str.endswith(prefix))

    def contains(self, pattern: str, *, literal: bool) -> DaftExpr:
        if not literal:
            return self.compliant._with_elementwise(lambda expr: expr.str.match(pattern))
        return self.compliant._with_elementwise(lambda expr: expr.str.contains(pattern))

    def split(self, by: str) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.str.split(by))

    def len_chars(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.str.length())

    def to_date(self, format: str | None) -> DaftExpr:
        if format is None:
            return self.compliant._with_elementwise(lambda expr: expr.cast("date"))
        return self.compliant._with_elementwise(lambda expr: expr.str.to_date(format))

    def to_datetime(self, format: str | None) -> DaftExpr:
        if format is None:
            msg = "`format` must be specified for Daft in `to_date`."
            raise ValueError(msg)
        return self.compliant._with_elementwise(
            lambda expr: expr.str.to_datetime(format).date()
        )

    def to_lowercase(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.str.lower())

    def to_uppercase(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.str.upper())

    def strip_chars(self, characters: str | None) -> DaftExpr:
        if characters is None:
            return self.compliant._with_elementwise(
                lambda expr: expr.str.lstrip().str.rstrip()
            )
        msg = "`strip_chars` with `characters` is currently not supported for Daft"
        raise NotImplementedError(msg)

    def replace(self, pattern: str, value: str, *, literal: bool, n: int) -> DaftExpr:
        msg = "`replace` is currently not supported for Daft"
        raise NotImplementedError(msg)

    def replace_all(self, pattern: str, value: str, *, literal: bool) -> DaftExpr:
        return self.compliant._with_elementwise(
            lambda expr: expr.str.replace(pattern, value, regex=not literal)
        )

    def slice(self, offset: int, length: int | None) -> DaftExpr:
        offset_lit = daft.lit(offset).cast("uint64")

        def func(expr: Expression) -> Expression:
            length_expr = expr.str.length() if length is None else daft.lit(length)
            offset_expr = (
                expr.str.length() + offset_lit
                if offset < 0
                else daft.lit(offset).cast("uint64")
            )
            return expr.str.substr(offset_expr, length_expr)

        return self.compliant._with_elementwise(func)
