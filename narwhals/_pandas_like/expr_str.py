from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.expr import PandasLikeExpr


class PandasLikeExprStringNamespace:
    def __init__(self: Self, expr: PandasLikeExpr) -> None:
        self._compliant_expr = expr

    def len_chars(self: Self) -> PandasLikeExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "str", "len_chars"
        )

    def replace(
        self: Self, pattern: str, value: str, *, literal: bool, n: int
    ) -> PandasLikeExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "str", "replace", pattern=pattern, value=value, literal=literal, n=n
        )

    def replace_all(
        self: Self, pattern: str, value: str, *, literal: bool
    ) -> PandasLikeExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "str", "replace_all", pattern=pattern, value=value, literal=literal
        )

    def strip_chars(self: Self, characters: str | None) -> PandasLikeExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "str", "strip_chars", characters=characters
        )

    def starts_with(self: Self, prefix: str) -> PandasLikeExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "str", "starts_with", prefix=prefix
        )

    def ends_with(self: Self, suffix: str) -> PandasLikeExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "str", "ends_with", suffix=suffix
        )

    def contains(self: Self, pattern: str, *, literal: bool) -> PandasLikeExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "str", "contains", pattern=pattern, literal=literal
        )

    def slice(self: Self, offset: int, length: int | None) -> PandasLikeExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "str", "slice", offset=offset, length=length
        )

    def split(self: Self, by: str) -> PandasLikeExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "str", "split", by=by
        )

    def to_datetime(self: Self, format: str | None) -> PandasLikeExpr:  # noqa: A002
        return self._compliant_expr._reuse_series_namespace_implementation(
            "str", "to_datetime", format=format
        )

    def to_uppercase(self: Self) -> PandasLikeExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "str", "to_uppercase"
        )

    def to_lowercase(self: Self) -> PandasLikeExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "str", "to_lowercase"
        )
