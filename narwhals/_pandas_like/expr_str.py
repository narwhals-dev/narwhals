from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._expression_parsing import reuse_series_namespace_implementation

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.expr import PandasLikeExpr


class PandasLikeExprStringNamespace:
    def __init__(self, expr: PandasLikeExpr) -> None:
        self._compliant_expr = expr

    def len_chars(
        self,
    ) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._compliant_expr, "str", "len_chars"
        )

    def replace(
        self,
        pattern: str,
        value: str,
        *,
        literal: bool = False,
        n: int = 1,
    ) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._compliant_expr,
            "str",
            "replace",
            pattern=pattern,
            value=value,
            literal=literal,
            n=n,
        )

    def replace_all(
        self,
        pattern: str,
        value: str,
        *,
        literal: bool = False,
    ) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._compliant_expr,
            "str",
            "replace_all",
            pattern=pattern,
            value=value,
            literal=literal,
        )

    def strip_chars(self, characters: str | None = None) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._compliant_expr,
            "str",
            "strip_chars",
            characters=characters,
        )

    def starts_with(self, prefix: str) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._compliant_expr,
            "str",
            "starts_with",
            prefix=prefix,
        )

    def ends_with(self, suffix: str) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._compliant_expr,
            "str",
            "ends_with",
            suffix=suffix,
        )

    def contains(self, pattern: str, *, literal: bool) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._compliant_expr,
            "str",
            "contains",
            pattern=pattern,
            literal=literal,
        )

    def slice(self, offset: int, length: int | None = None) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._compliant_expr, "str", "slice", offset=offset, length=length
        )

    def to_datetime(self: Self, format: str | None) -> PandasLikeExpr:  # noqa: A002
        return reuse_series_namespace_implementation(
            self._compliant_expr,
            "str",
            "to_datetime",
            format=format,
        )

    def to_uppercase(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._compliant_expr,
            "str",
            "to_uppercase",
        )

    def to_lowercase(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._compliant_expr,
            "str",
            "to_lowercase",
        )