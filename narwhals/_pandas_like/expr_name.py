from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from narwhals.exceptions import AnonymousExprError

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.expr import PandasLikeExpr


class PandasLikeExprNameNamespace:
    def __init__(self: Self, expr: PandasLikeExpr) -> None:
        self._compliant_expr = expr

    def keep(self: Self) -> PandasLikeExpr:
        root_names = self._compliant_expr._root_names

        if root_names is None:
            msg = ".name.keep"
            raise AnonymousExprError.from_expr_name(msg)

        return self._compliant_expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(self._compliant_expr._call(df), root_names)
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            root_names=root_names,
            output_names=root_names,
            implementation=self._compliant_expr._implementation,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs=self._compliant_expr._kwargs,
        )

    def map(self: Self, function: Callable[[str], str]) -> PandasLikeExpr:
        root_names = self._compliant_expr._root_names

        if root_names is None:
            msg = ".name.map"
            raise AnonymousExprError.from_expr_name(msg)

        output_names = [function(str(name)) for name in root_names]

        return self._compliant_expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(self._compliant_expr._call(df), output_names)
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._compliant_expr._implementation,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs={**self._compliant_expr._kwargs, "function": function},
        )

    def prefix(self: Self, prefix: str) -> PandasLikeExpr:
        root_names = self._compliant_expr._root_names
        if root_names is None:
            msg = ".name.prefix"
            raise AnonymousExprError.from_expr_name(msg)

        output_names = [prefix + str(name) for name in root_names]
        return self._compliant_expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(self._compliant_expr._call(df), output_names)
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._compliant_expr._implementation,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs={**self._compliant_expr._kwargs, "prefix": prefix},
        )

    def suffix(self: Self, suffix: str) -> PandasLikeExpr:
        root_names = self._compliant_expr._root_names
        if root_names is None:
            msg = ".name.suffix"
            raise AnonymousExprError.from_expr_name(msg)

        output_names = [str(name) + suffix for name in root_names]

        return self._compliant_expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(self._compliant_expr._call(df), output_names)
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._compliant_expr._implementation,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs={**self._compliant_expr._kwargs, "suffix": suffix},
        )

    def to_lowercase(self: Self) -> PandasLikeExpr:
        root_names = self._compliant_expr._root_names

        if root_names is None:
            msg = ".name.to_lowercase"
            raise AnonymousExprError.from_expr_name(msg)

        output_names = [str(name).lower() for name in root_names]

        return self._compliant_expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(self._compliant_expr._call(df), output_names)
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._compliant_expr._implementation,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs=self._compliant_expr._kwargs,
        )

    def to_uppercase(self: Self) -> PandasLikeExpr:
        root_names = self._compliant_expr._root_names

        if root_names is None:
            msg = ".name.to_uppercase"
            raise AnonymousExprError.from_expr_name(msg)

        output_names = [str(name).upper() for name in root_names]

        return self._compliant_expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(self._compliant_expr._call(df), output_names)
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._compliant_expr._implementation,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs=self._compliant_expr._kwargs,
        )
