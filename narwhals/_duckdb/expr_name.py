from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprNameNamespace:
    def __init__(self: Self, expr: DuckDBExpr) -> None:
        self._compliant_expr = expr

    def keep(self: Self) -> DuckDBExpr:
        return self._compliant_expr.__class__(
            self._compliant_expr._call,
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=None,
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
        )

    def map(self: Self, function: Callable[[str], str]) -> DuckDBExpr:
        return self._compliant_expr.__class__(
            self._compliant_expr._call,
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=lambda output_names: [
                function(str(name)) for name in output_names
            ],
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
        )

    def prefix(self: Self, prefix: str) -> DuckDBExpr:
        return self._compliant_expr.__class__(
            self._compliant_expr._call,
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=lambda output_names: [
                f"{prefix}{output_name}" for output_name in output_names
            ],
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
        )

    def suffix(self: Self, suffix: str) -> DuckDBExpr:
        return self._compliant_expr.__class__(
            self._compliant_expr._call,
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=lambda output_names: [
                f"{output_name}{suffix}" for output_name in output_names
            ],
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
        )

    def to_lowercase(self: Self) -> DuckDBExpr:
        return self._compliant_expr.__class__(
            self._compliant_expr._call,
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=lambda output_names: [
                str(name).lower() for name in output_names
            ],
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
        )

    def to_uppercase(self: Self) -> DuckDBExpr:
        return self._compliant_expr.__class__(
            self._compliant_expr._call,
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=lambda output_names: [
                str(name).upper() for name in output_names
            ],
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
        )
