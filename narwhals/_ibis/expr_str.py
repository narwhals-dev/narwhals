from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ibis.expr.datatypes import Timestamp

from narwhals.utils import _is_naive_format, not_implemented

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from narwhals._ibis.expr import IbisExpr


class IbisExprStringNamespace:
    def __init__(self, expr: IbisExpr) -> None:
        self._compliant_expr = expr

    def starts_with(self, prefix: str) -> IbisExpr:
        def fn(expr: ir.StringColumn) -> ir.BooleanValue:
            return expr.startswith(prefix)

        return self._compliant_expr._with_callable(fn)

    def ends_with(self, suffix: str) -> IbisExpr:
        def fn(expr: ir.StringColumn) -> ir.BooleanValue:
            return expr.endswith(suffix)

        return self._compliant_expr._with_callable(fn)

    def contains(self, pattern: str, *, literal: bool) -> IbisExpr:
        def fn(expr: ir.StringColumn) -> ir.BooleanValue:
            return expr.contains(pattern) if literal else expr.re_search(pattern)

        return self._compliant_expr._with_callable(fn)

    def slice(self, offset: int, length: int) -> IbisExpr:
        def fn(expr: ir.StringColumn) -> ir.StringValue:
            return expr.substr(start=offset, length=length)

        return self._compliant_expr._with_callable(fn)

    def split(self, by: str) -> IbisExpr:
        def fn(expr: ir.StringColumn) -> ir.ArrayValue:
            return expr.split(by)

        return self._compliant_expr._with_callable(fn)

    def len_chars(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.length())

    def to_lowercase(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.lower())

    def to_uppercase(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.upper())

    def strip_chars(self, characters: str | None) -> IbisExpr:
        if characters is not None:
            msg = "Ibis does not support `characters` argument in `str.strip_chars`"
            raise NotImplementedError(msg)

        return self._compliant_expr._with_callable(lambda expr: expr.strip())

    def _replace_all(self, pattern: str, value: str) -> Callable[..., ir.StringValue]:
        def fn(expr: ir.StringColumn) -> ir.StringValue:
            return expr.re_replace(pattern, value)

        return fn

    def _replace_all_literal(
        self, pattern: str, value: str
    ) -> Callable[..., ir.StringValue]:
        def fn(expr: ir.StringColumn) -> ir.StringValue:
            return expr.replace(pattern, value)  # pyright: ignore[reportArgumentType]

        return fn

    def replace_all(self, pattern: str, value: str, *, literal: bool) -> IbisExpr:
        fn = self._replace_all_literal if literal else self._replace_all
        return self._compliant_expr._with_callable(fn(pattern, value))

    def _to_datetime(self, format: str) -> Callable[..., ir.TimestampValue]:
        def fn(expr: ir.StringColumn) -> ir.TimestampValue:
            return expr.as_timestamp(format)

        return fn

    def _to_datetime_naive(self, format: str) -> Callable[..., ir.TimestampValue]:
        def fn(expr: ir.StringColumn) -> ir.TimestampValue:
            dtype: Any = Timestamp(timezone=None)
            return expr.as_timestamp(format).cast(dtype)

        return fn

    def to_datetime(self, format: str | None) -> IbisExpr:
        if format is None:
            msg = "Cannot infer format with Ibis backend"
            raise NotImplementedError(msg)
        fn = self._to_datetime_naive if _is_naive_format(format) else self._to_datetime
        return self._compliant_expr._with_callable(fn(format))

    replace = not_implemented()
