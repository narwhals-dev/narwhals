from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from ibis.expr.datatypes import Timestamp

from narwhals.utils import not_implemented

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from narwhals._ibis.expr import IbisExpr


class IbisExprStringNamespace:
    def __init__(self, expr: IbisExpr) -> None:
        self._compliant_expr = expr

    def starts_with(self, prefix: str) -> IbisExpr:
        def fn(_input: ir.StringColumn) -> ir.BooleanValue:
            return _input.startswith(prefix)

        return self._compliant_expr._with_callable(fn)

    def ends_with(self, suffix: str) -> IbisExpr:
        def fn(_input: ir.StringColumn) -> ir.BooleanValue:
            return _input.endswith(suffix)

        return self._compliant_expr._with_callable(fn)

    def contains(self, pattern: str, *, literal: bool) -> IbisExpr:
        def fn(_input: ir.StringColumn) -> ir.BooleanValue:
            return _input.contains(pattern) if literal else _input.re_search(pattern)

        return self._compliant_expr._with_callable(fn)

    def slice(self, offset: int, length: int) -> IbisExpr:
        def fn(_input: ir.StringColumn) -> ir.StringValue:
            return _input.substr(start=offset, length=length)

        return self._compliant_expr._with_callable(fn)

    def split(self, by: str) -> IbisExpr:
        def fn(_input: ir.StringColumn) -> ir.ArrayValue:
            return _input.split(by)

        return self._compliant_expr._with_callable(fn)

    def len_chars(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.length())

    def to_lowercase(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.lower())

    def to_uppercase(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.upper())

    def strip_chars(self, characters: str | None) -> IbisExpr:
        if characters is not None:
            msg = "Ibis does not support `characters` argument in `str.strip_chars`"
            raise NotImplementedError(msg)

        return self._compliant_expr._with_callable(lambda _input: _input.strip())

    def _replace_all(self, pattern: str, value: str) -> Callable[..., ir.StringValue]:
        def fn(_input: ir.StringColumn) -> ir.StringValue:
            return _input.re_replace(pattern, value)

        return fn

    def _replace_all_literal(
        self, pattern: str, value: str
    ) -> Callable[..., ir.StringValue]:
        def fn(_input: ir.StringColumn) -> ir.StringValue:
            return _input.replace(pattern, value)  # pyright: ignore[reportArgumentType]

        return fn

    def replace_all(self, pattern: str, value: str, *, literal: bool) -> IbisExpr:
        fn = self._replace_all_literal if literal else self._replace_all
        return self._compliant_expr._with_callable(fn(pattern, value))

    def _to_datetime(self, format: str) -> Callable[..., ir.TimestampValue]:
        def fn(_input: ir.StringColumn) -> ir.TimestampValue:
            return _input.as_timestamp(format)

        return fn

    def _to_datetime_naive(self, format: str) -> Callable[..., ir.TimestampValue]:
        def fn(_input: ir.StringColumn) -> ir.TimestampValue:
            dtype: Any = Timestamp(timezone=None)
            return _input.as_timestamp(format).cast(dtype)

        return fn

    def to_datetime(self, format: str | None) -> IbisExpr:
        if format is None:
            msg = "Cannot infer format with Ibis backend"
            raise NotImplementedError(msg)
        fn = self._to_datetime_naive if _is_naive_format(format) else self._to_datetime
        return self._compliant_expr._with_callable(fn(format))

    replace = not_implemented()


def _is_naive_format(format_: str) -> bool:
    return not any(x in format_ for x in ("%s", "%z", "Z"))
