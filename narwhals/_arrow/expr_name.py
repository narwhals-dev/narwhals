from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.expr import ArrowExpr


class ArrowExprNameNamespace:
    def __init__(self: Self, expr: ArrowExpr) -> None:
        self._compliant_expr = expr

    def keep(self: Self) -> ArrowExpr:
        return self._compliant_expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(
                    self._compliant_expr._call(df),
                    self._compliant_expr._evaluate_output_names(df),
                )
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=None,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs=self._compliant_expr._kwargs,
        )

    def map(self: Self, function: Callable[[str], str]) -> ArrowExpr:
        return self._compliant_expr.__class__(
            lambda df: [
                series.alias(function(str(name)))
                for series, name in zip(
                    self._compliant_expr._call(df),
                    self._compliant_expr._evaluate_output_names(df),
                )
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=lambda output_names: [
                function(str(name)) for name in output_names
            ],
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs={**self._compliant_expr._kwargs, "function": function},
        )

    def prefix(self: Self, prefix: str) -> ArrowExpr:
        return self._compliant_expr.__class__(
            lambda df: [
                series.alias(f"{prefix}{name}")
                for series, name in zip(
                    self._compliant_expr._call(df),
                    self._compliant_expr._evaluate_output_names(df),
                )
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=lambda output_names: [
                f"{prefix}{output_name}" for output_name in output_names
            ],
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs={**self._compliant_expr._kwargs, "prefix": prefix},
        )

    def suffix(self: Self, suffix: str) -> ArrowExpr:
        return self._compliant_expr.__class__(
            lambda df: [
                series.alias(f"{name}{suffix}")
                for series, name in zip(
                    self._compliant_expr._call(df),
                    self._compliant_expr._evaluate_output_names(df),
                )
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=lambda output_names: [
                f"{output_name}{suffix}" for output_name in output_names
            ],
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs={**self._compliant_expr._kwargs, "suffix": suffix},
        )

    def to_lowercase(self: Self) -> ArrowExpr:
        return self._compliant_expr.__class__(
            lambda df: [
                series.alias(str(name).lower())
                for series, name in zip(
                    self._compliant_expr._call(df),
                    self._compliant_expr._evaluate_output_names(df),
                )
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=lambda output_names: [
                str(name).lower() for name in output_names
            ],
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs=self._compliant_expr._kwargs,
        )

    def to_uppercase(self: Self) -> ArrowExpr:
        return self._compliant_expr.__class__(
            lambda df: [
                series.alias(str(name).upper())
                for series, name in zip(
                    self._compliant_expr._call(df),
                    self._compliant_expr._evaluate_output_names(df),
                )
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=lambda output_names: [
                str(name).upper() for name in output_names
            ],
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs=self._compliant_expr._kwargs,
        )
