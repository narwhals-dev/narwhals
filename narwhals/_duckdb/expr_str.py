from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NoReturn

from duckdb import ConstantExpression
from duckdb import FunctionExpression

if TYPE_CHECKING:
    import duckdb
    from typing_extensions import Self

    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprStringNamespace:
    def __init__(self: Self, expr: DuckDBExpr) -> None:
        self._compliant_expr = expr

    def starts_with(self: Self, prefix: str) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression(
                "starts_with", _input, ConstantExpression(prefix)
            ),
            "starts_with",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def ends_with(self: Self, suffix: str) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression(
                "ends_with", _input, ConstantExpression(suffix)
            ),
            "ends_with",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def contains(self: Self, pattern: str, *, literal: bool) -> DuckDBExpr:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            if literal:
                return FunctionExpression("contains", _input, ConstantExpression(pattern))
            return FunctionExpression(
                "regexp_matches", _input, ConstantExpression(pattern)
            )

        return self._compliant_expr._from_call(
            func, "contains", returns_scalar=self._compliant_expr._returns_scalar
        )

    def slice(self: Self, offset: int, length: int) -> DuckDBExpr:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            return FunctionExpression(
                "array_slice",
                _input,
                ConstantExpression(offset + 1)
                if offset >= 0
                else FunctionExpression("length", _input) + offset + 1,
                FunctionExpression("length", _input)
                if length is None
                else ConstantExpression(length) + offset,
            )

        return self._compliant_expr._from_call(
            func, "slice", returns_scalar=self._compliant_expr._returns_scalar
        )

    def len_chars(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("length", _input),
            "len_chars",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def to_lowercase(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("lower", _input),
            "to_lowercase",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def to_uppercase(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("upper", _input),
            "to_uppercase",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def strip_chars(self: Self, characters: str | None) -> DuckDBExpr:
        import string

        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression(
                "trim",
                _input,
                ConstantExpression(
                    string.whitespace if characters is None else characters
                ),
            ),
            "strip_chars",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def replace_all(self: Self, pattern: str, value: str, *, literal: bool) -> DuckDBExpr:
        if literal is False:
            return self._compliant_expr._from_call(
                lambda _input: FunctionExpression(
                    "regexp_replace",
                    _input,
                    ConstantExpression(pattern),
                    ConstantExpression(value),
                    ConstantExpression("g"),
                ),
                "replace_all",
                returns_scalar=self._compliant_expr._returns_scalar,
            )
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression(
                "replace", _input, ConstantExpression(pattern), ConstantExpression(value)
            ),
            "replace_all",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def replace(
        self: Self, pattern: str, value: str, *, literal: bool, n: int
    ) -> NoReturn:
        msg = "`replace` is currently not supported for DuckDB"
        raise NotImplementedError(msg)
