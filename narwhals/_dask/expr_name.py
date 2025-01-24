from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from narwhals.exceptions import AnonymousExprError

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._dask.expr import DaskExpr


class DaskExprNameNamespace:
    def __init__(self: Self, expr: DaskExpr) -> None:
        self._compliant_expr = expr

    def keep(self: Self) -> DaskExpr:
        return self._compliant_expr.__class__(
            self._compliant_expr._call,
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_root_names=self._compliant_expr._evaluate_root_names,
            evaluate_aliases=None,
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs=self._compliant_expr._kwargs,
        )

    def map(self: Self, function: Callable[[str], str]) -> DaskExpr:
        return self._compliant_expr.__class__(
            self._compliant_expr._call,
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_root_names=self._compliant_expr._evaluate_root_names,
            evaluate_aliases=lambda root_names: [function(str(name)) for name in root_names],
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs={**self._compliant_expr._kwargs, "function": function},
        )

    def prefix(self: Self, prefix: str) -> DaskExpr:
        return self._compliant_expr.__class__(
            self._compliant_expr._call,
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_root_names=self._compliant_expr._evaluate_root_names,
            evaluate_aliases=lambda root_names: [f'{prefix}{root_name}' for root_name in root_names],
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs={**self._compliant_expr._kwargs, "prefix": prefix},
        )

    def suffix(self: Self, suffix: str) -> DaskExpr:
        return self._compliant_expr.__class__(
            self._compliant_expr._call,
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_root_names=self._compliant_expr._evaluate_root_names,
            evaluate_aliases=lambda root_names: [f'{root_name}{suffix}' for root_name in root_names],
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs={**self._compliant_expr._kwargs, "suffix": suffix},
        )

    def to_lowercase(self: Self) -> DaskExpr:
        return self._compliant_expr.__class__(
            self._compliant_expr._call,
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_root_names=self._compliant_expr._evaluate_root_names,
            evaluate_aliases=lambda root_names: [str(name).lower() for name in root_names],
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs=self._compliant_expr._kwargs,
        )

    def to_uppercase(self: Self) -> DaskExpr:
        return self._compliant_expr.__class__(
            self._compliant_expr._call,
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_root_names=self._compliant_expr._evaluate_root_names,
            evaluate_aliases=lambda root_names: [str(name).upper() for name in root_names],
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs=self._compliant_expr._kwargs,
        )
