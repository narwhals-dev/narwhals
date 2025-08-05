from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._duckdb.utils import F, lit
from narwhals._sql.expr_str import SQLExprStringNamespace
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from duckdb import Expression

    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprStringNamespace(SQLExprStringNamespace["DuckDBExpr"]):
    def contains(self, pattern: str, *, literal: bool) -> DuckDBExpr:
        def func(expr: Expression) -> Expression:
            if literal:
                return F("contains", expr, lit(pattern))
            return F("regexp_matches", expr, lit(pattern))

        return self.compliant._with_elementwise(func)

    def slice(self, offset: int, length: int | None) -> DuckDBExpr:
        def func(expr: Expression) -> Expression:
            offset_lit = lit(offset)
            return F(
                "array_slice",
                expr,
                lit(offset + 1)
                if offset >= 0
                else F("length", expr) + offset_lit + lit(1),
                F("length", expr) if length is None else lit(length) + offset_lit,
            )

        return self.compliant._with_elementwise(func)

    def split(self, by: str) -> DuckDBExpr:
        return self.compliant._with_elementwise(
            lambda expr: F("str_split", expr, lit(by))
        )

    def len_chars(self) -> DuckDBExpr:
        return self.compliant._with_elementwise(lambda expr: F("length", expr))

    def strip_chars(self, characters: str | None) -> DuckDBExpr:
        import string

        return self.compliant._with_elementwise(
            lambda expr: F(
                "trim", expr, lit(string.whitespace if characters is None else characters)
            )
        )

    def replace_all(self, pattern: str, value: str, *, literal: bool) -> DuckDBExpr:
        if not literal:
            return self.compliant._with_elementwise(
                lambda expr: F("regexp_replace", expr, lit(pattern), lit(value), lit("g"))
            )
        return self.compliant._with_elementwise(
            lambda expr: F("replace", expr, lit(pattern), lit(value))
        )

    def to_datetime(self, format: str | None) -> DuckDBExpr:
        if format is None:
            msg = "Cannot infer format with DuckDB backend, please specify `format` explicitly."
            raise NotImplementedError(msg)

        return self.compliant._with_elementwise(
            lambda expr: F("strptime", expr, lit(format))
        )

    def to_date(self, format: str | None) -> DuckDBExpr:
        if format is not None:
            return self.to_datetime(format=format).dt.date()

        compliant_expr = self.compliant
        return compliant_expr.cast(compliant_expr._version.dtypes.Date())

    replace = not_implemented()
