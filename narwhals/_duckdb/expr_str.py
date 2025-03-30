from __future__ import annotations

from typing import TYPE_CHECKING

from duckdb import FunctionExpression

from narwhals._duckdb.utils import lit

if TYPE_CHECKING:
    import duckdb
    from typing_extensions import Never
    from typing_extensions import Self

    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprStringNamespace:
    def __init__(self: Self, expr: DuckDBExpr) -> None:
        self._compliant_expr = expr

    def starts_with(self: Self, prefix: str) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("starts_with", _input, lit(prefix))
        )

    def ends_with(self: Self, suffix: str) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("ends_with", _input, lit(suffix))
        )

    def contains(self: Self, pattern: str, *, literal: bool) -> DuckDBExpr:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            if literal:
                return FunctionExpression("contains", _input, lit(pattern))
            return FunctionExpression("regexp_matches", _input, lit(pattern))

        return self._compliant_expr._with_callable(func)

    def slice(self: Self, offset: int, length: int) -> DuckDBExpr:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            offset_lit = lit(offset)
            return FunctionExpression(
                "array_slice",
                _input,
                lit(offset + 1)
                if offset >= 0
                else FunctionExpression("length", _input) + offset_lit + lit(1),
                FunctionExpression("length", _input)
                if length is None
                else lit(length) + offset_lit,
            )

        return self._compliant_expr._with_callable(func)

    def split(self: Self, by: str) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("str_split", _input, lit(by))
        )

    def len_chars(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("length", _input)
        )

    def to_lowercase(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("lower", _input)
        )

    def to_uppercase(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("upper", _input)
        )

    def strip_chars(self: Self, characters: str | None) -> DuckDBExpr:
        import string

        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression(
                "trim",
                _input,
                lit(string.whitespace if characters is None else characters),
            )
        )

    def replace_all(self: Self, pattern: str, value: str, *, literal: bool) -> DuckDBExpr:
        if not literal:
            return self._compliant_expr._with_callable(
                lambda _input: FunctionExpression(
                    "regexp_replace", _input, lit(pattern), lit(value), lit("g")
                )
            )
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("replace", _input, lit(pattern), lit(value))
        )

    def replace(self: Self, pattern: str, value: str, *, literal: bool, n: int) -> Never:
        msg = "`replace` is currently not supported for DuckDB"
        raise NotImplementedError(msg)

    def to_datetime(self: Self, format: str | None) -> DuckDBExpr:
        if format is None:
            msg = "Cannot infer format with DuckDB backend, please specify `format` explicitly."
            raise NotImplementedError(msg)

        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("strptime", _input, lit(format))
        )
