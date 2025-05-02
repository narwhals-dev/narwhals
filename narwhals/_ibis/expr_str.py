from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals.utils import not_implemented

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from narwhals._ibis.expr import IbisExpr


class IbisExprStringNamespace:
    def __init__(self, expr: IbisExpr) -> None:
        self._compliant_expr = expr

    def starts_with(self, prefix: str) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.startswith(prefix),
        )

    def ends_with(self, suffix: str) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.endswith(suffix),
        )

    def contains(self, pattern: str, *, literal: bool) -> IbisExpr:
        def func(_input: ir.StringColumn) -> ir.BooleanValue:
            if literal:
                return _input.contains(pattern)
            return _input.re_search(pattern)

        return self._compliant_expr._with_callable(func)

    def slice(self, offset: int, length: int) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.substr(start=offset, length=length),
        )

    def split(self, by: str) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.split(by))

    def len_chars(self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.length(),
        )

    def to_lowercase(self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.lower(),
        )

    def to_uppercase(self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.upper(),
        )

    def strip_chars(self, characters: str | None) -> IbisExpr:
        if characters is not None:
            msg = "Ibis does not support `characters` argument in `str.strip_chars`"
            raise NotImplementedError(msg)

        return self._compliant_expr._with_callable(
            lambda _input: _input.strip(),
        )

    def replace_all(self, pattern: str, value: str, *, literal: bool) -> IbisExpr:
        if not literal:
            return self._compliant_expr._with_callable(
                lambda _input: _input.re_replace(pattern, value),
            )
        return self._compliant_expr._with_callable(
            lambda _input: _input.replace(pattern, value),
        )

    def to_datetime(self, format: str | None) -> IbisExpr:
        from ibis.expr.datatypes import Timestamp

        if format is None:
            msg = "Cannot infer format with Ibis backend"
            raise NotImplementedError(msg)

        if _is_naive_format(format):
            return self._compliant_expr._with_callable(
                lambda _input: _input.as_timestamp(format).cast(Timestamp(timezone=None)),
            )
        else:
            return self._compliant_expr._with_callable(
                lambda _input: _input.as_timestamp(format),
            )

    replace = not_implemented()


def _is_naive_format(format_: str) -> bool:
    return not any(x in format_ for x in ("%s", "%z", "Z"))
