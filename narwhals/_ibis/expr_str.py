from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from typing_extensions import Never
    from typing_extensions import Self

    from narwhals._ibis.expr import IbisExpr


class IbisExprStringNamespace:
    def __init__(self: Self, expr: IbisExpr) -> None:
        self._compliant_expr = expr

    def starts_with(self: Self, prefix: str) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.startswith(prefix),
        )

    def ends_with(self: Self, suffix: str) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.endswith(suffix),
        )

    def contains(self: Self, pattern: str, *, literal: bool) -> IbisExpr:
        def func(_input: ir.Expr) -> ir.Expr:
            if literal:
                return _input.contains(pattern)
            return _input.re_search(pattern)

        return self._compliant_expr._with_callable(func)

    def slice(self: Self, offset: int, length: int) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.substr(start=offset, length=length),
        )

    def split(self: Self, by: str) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.split(by))

    def len_chars(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.length(),
        )

    def to_lowercase(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.lower(),
        )

    def to_uppercase(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.upper(),
        )

    def strip_chars(self: Self, characters: str | None) -> IbisExpr:
        if characters is not None:
            msg = "Ibis does not support `characters` argument in `str.strip_chars`"
            raise NotImplementedError(msg)

        return self._compliant_expr._with_callable(
            lambda _input: _input.strip(),
        )

    def replace_all(self: Self, pattern: str, value: str, *, literal: bool) -> IbisExpr:
        if not literal:
            return self._compliant_expr._with_callable(
                lambda _input: _input.re_replace(pattern, value),
            )
        return self._compliant_expr._with_callable(
            lambda _input: _input.replace(pattern, value),
        )

    def replace(self: Self, pattern: str, value: str, *, literal: bool, n: int) -> Never:
        msg = "`replace` is currently not supported for Ibis"
        raise NotImplementedError(msg)

    def to_datetime(self: Self, format: str | None) -> IbisExpr:
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


def _is_naive_format(format_: str) -> bool:
    return not any(x in format_ for x in ("%s", "%z", "Z"))
