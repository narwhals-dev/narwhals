from __future__ import annotations

from typing import TYPE_CHECKING

from duckdb import FunctionExpression

from narwhals._duckdb.utils import lit, when
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from duckdb import Expression

    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprStringNamespace:
    def __init__(self, expr: DuckDBExpr) -> None:
        self._compliant_expr = expr

    def starts_with(self, prefix: str) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("starts_with", expr, lit(prefix))
        )

    def ends_with(self, suffix: str) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("ends_with", expr, lit(suffix))
        )

    def contains(self, pattern: str, *, literal: bool) -> DuckDBExpr:
        def func(expr: Expression) -> Expression:
            if literal:
                return FunctionExpression("contains", expr, lit(pattern))
            return FunctionExpression("regexp_matches", expr, lit(pattern))

        return self._compliant_expr._with_callable(func)

    def slice(self, offset: int, length: int) -> DuckDBExpr:
        def func(expr: Expression) -> Expression:
            offset_lit = lit(offset)
            return FunctionExpression(
                "array_slice",
                expr,
                lit(offset + 1)
                if offset >= 0
                else FunctionExpression("length", expr) + offset_lit + lit(1),
                FunctionExpression("length", expr)
                if length is None
                else lit(length) + offset_lit,
            )

        return self._compliant_expr._with_callable(func)

    def split(self, by: str) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("str_split", expr, lit(by))
        )

    def len_chars(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("length", expr)
        )

    def to_lowercase(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("lower", expr)
        )

    def to_uppercase(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("upper", expr)
        )

    def strip_chars(self, characters: str | None) -> DuckDBExpr:
        import string

        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression(
                "trim", expr, lit(string.whitespace if characters is None else characters)
            )
        )

    def replace_all(self, pattern: str, value: str, *, literal: bool) -> DuckDBExpr:
        if not literal:
            return self._compliant_expr._with_callable(
                lambda expr: FunctionExpression(
                    "regexp_replace", expr, lit(pattern), lit(value), lit("g")
                )
            )
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("replace", expr, lit(pattern), lit(value))
        )

    def to_datetime(self, format: str | None) -> DuckDBExpr:
        if format is None:
            msg = "Cannot infer format with DuckDB backend, please specify `format` explicitly."
            raise NotImplementedError(msg)

        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("strptime", expr, lit(format))
        )

    def zfill(self, width: int) -> DuckDBExpr:
        # DuckDB does not have a built-in zfill function, so we need to implement it manually
        # using string manipulation functions.

        def func(expr: Expression) -> Expression:
            less_than_width = FunctionExpression("length", expr) < lit(width)
            zero, hyphen, plus = lit("0"), lit("-"), lit("+")

            starts_with_minus = FunctionExpression("starts_with", expr, hyphen)
            starts_with_plus = FunctionExpression("starts_with", expr, plus)
            substring = FunctionExpression("substr", expr, lit(2))
            padded_substring = FunctionExpression("lpad", substring, lit(width - 1), zero)
            return (
                when(
                    starts_with_minus & less_than_width,
                    FunctionExpression("concat", hyphen, padded_substring),
                )
                .when(
                    starts_with_plus & less_than_width,
                    FunctionExpression("concat", plus, padded_substring),
                )
                .when(less_than_width, FunctionExpression("lpad", expr, lit(width), zero))
                .otherwise(expr)
            )

        return self._compliant_expr._with_callable(func)

    replace = not_implemented()
