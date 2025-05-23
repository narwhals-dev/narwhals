from __future__ import annotations

from typing import TYPE_CHECKING

import dask.dataframe as dd

if TYPE_CHECKING:
    from narwhals._dask.expr import DaskExpr


class DaskExprStringNamespace:
    def __init__(self, expr: DaskExpr) -> None:
        self._compliant_expr = expr

    def len_chars(self) -> DaskExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.str.len(), "len")

    def replace(self, pattern: str, value: str, *, literal: bool, n: int) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda expr, pattern, value, literal, n: expr.str.replace(
                pattern, value, regex=not literal, n=n
            ),
            "replace",
            pattern=pattern,
            value=value,
            literal=literal,
            n=n,
        )

    def replace_all(self, pattern: str, value: str, *, literal: bool) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda expr, pattern, value, literal: expr.str.replace(
                pattern, value, n=-1, regex=not literal
            ),
            "replace",
            pattern=pattern,
            value=value,
            literal=literal,
        )

    def strip_chars(self, characters: str | None) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda expr, characters: expr.str.strip(characters),
            "strip",
            characters=characters,
        )

    def starts_with(self, prefix: str) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda expr, prefix: expr.str.startswith(prefix), "starts_with", prefix=prefix
        )

    def ends_with(self, suffix: str) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda expr, suffix: expr.str.endswith(suffix), "ends_with", suffix=suffix
        )

    def contains(self, pattern: str, *, literal: bool) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda expr, pattern, literal: expr.str.contains(
                pat=pattern, regex=not literal
            ),
            "contains",
            pattern=pattern,
            literal=literal,
        )

    def slice(self, offset: int, length: int | None) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda expr, offset, length: expr.str.slice(
                start=offset, stop=offset + length if length else None
            ),
            "slice",
            offset=offset,
            length=length,
        )

    def split(self, by: str) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda expr, by: expr.str.split(pat=by), "split", by=by
        )

    def to_datetime(self, format: str | None) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda expr, format: dd.to_datetime(expr, format=format),
            "to_datetime",
            format=format,
        )

    def to_uppercase(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda expr: expr.str.upper(), "to_uppercase"
        )

    def to_lowercase(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda expr: expr.str.lower(), "to_lowercase"
        )
