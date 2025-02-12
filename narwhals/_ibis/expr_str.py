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
        return self._compliant_expr._from_call(
            lambda _input: _input.startswith(prefix),
            "starts_with",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def ends_with(self: Self, suffix: str) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.endswith(suffix),
            "ends_with",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def contains(self: Self, pattern: str, *, literal: bool) -> IbisExpr:
        def func(_input: ir.Expr) -> ir.Expr:
            if literal:
                return _input.contains(pattern)
            return _input.re_search(pattern)

        return self._compliant_expr._from_call(
            func, "contains", expr_kind=self._compliant_expr._expr_kind
        )

    def slice(self: Self, offset: int, length: int) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.substr(start=offset, length=length),
            "slice",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def len_chars(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.length(),
            "len_chars",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def to_lowercase(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.lower(),
            "to_lowercase",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def to_uppercase(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.upper(),
            "to_uppercase",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def strip_chars(self: Self, characters: str | None) -> IbisExpr:
        if characters is not None:
            msg = "Ibis does not support `characters` argument in `str.strip_chars`"
            raise NotImplementedError(msg)

        return self._compliant_expr._from_call(
            lambda _input: _input.strip(),
            "strip_chars",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def replace_all(self: Self, pattern: str, value: str, *, literal: bool) -> IbisExpr:
        if not literal:
            return self._compliant_expr._from_call(
                lambda _input: _input.re_replace(pattern, value),
                "replace_all",
                expr_kind=self._compliant_expr._expr_kind,
            )
        return self._compliant_expr._from_call(
            lambda _input: _input.replace(pattern, value),
            "replace_all",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def replace(self: Self, pattern: str, value: str, *, literal: bool, n: int) -> Never:
        msg = "`replace` is currently not supported for Ibis"
        raise NotImplementedError(msg)

    def to_datetime(self: Self, format: str | None) -> IbisExpr:  # noqa: A002
        from ibis.expr.datatypes import Timestamp

        if format is None:
            msg = "Cannot infer format with Ibis backend"
            raise NotImplementedError(msg)

        return self._compliant_expr._from_call(
            lambda _input: _input.as_timestamp(format).cast(Timestamp(timezone=None)),
            "to_datetime",
            expr_kind=self._compliant_expr._expr_kind,
        )
