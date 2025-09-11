"""A more forgiving `duckdb.FunctionExpression`."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain
from typing import TYPE_CHECKING, Any, Union

import duckdb
from duckdb import (  # noqa: N813
    ColumnExpression as _col,
    ConstantExpression as _lit,
    Expression,
    FunctionExpression as _F,
)

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias, TypeIs

    from narwhals._duckdb.typing import FunctionName, IntoNativeExpr
    from narwhals.typing import NonNestedLiteral

    IntoExpr: TypeAlias = Union[IntoNativeExpr, "_Expr"]

__all__ = ["F", "col", "lit"]


def _is_native_expr(obj: Any) -> TypeIs[duckdb.Expression]:
    return isinstance(obj, duckdb.Expression)


class _FExpr:
    def __getattr__(self, name: FunctionName) -> _FunctionExpressionCaller:  # type: ignore[misc]
        return _FunctionExpressionCaller(name)


class _FunctionExpressionCaller:
    def __init__(self, name: FunctionName, /) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"nw._duckdb.F.{self._name}"

    def __call__(self, *exprs: IntoExpr) -> _Expr:
        return _Expr.from_native(_F(self._name, *_into_native_expr_iter(exprs)))


class _FunctionExpressionMethodCaller:
    def __init__(self, name: FunctionName, /, expr: _Expr) -> None:
        self._name = name
        self._expr = expr

    def __repr__(self) -> str:
        return f"nw._duckdb.F.{self._name}({self._expr!r}, ...)"

    def __call__(self, *exprs: IntoExpr) -> _Expr:
        it = _into_native_expr_iter(exprs)
        return _Expr.from_native(_F(self._name, self._expr.native, *it))


F = _FExpr()
"""`duckdb.FunctionExpression` with a twist."""


class _Expr:
    _expr: Expression

    def __repr__(self) -> str:
        return f"nw._duckdb._Expr<{self.native!r}>"

    @classmethod
    def from_native(cls, expr: Expression, /) -> Self:
        obj = cls.__new__(cls)
        obj._expr = expr
        return obj

    def __getattr__(self, name: FunctionName) -> _FunctionExpressionMethodCaller:  # type: ignore[misc]
        return _FunctionExpressionMethodCaller(name, self)

    @classmethod
    def from_scalar(cls, value: NonNestedLiteral, /) -> Self:
        return cls.from_native(_lit(value))

    @classmethod
    def from_name(cls, column_name: str, /) -> Self:
        return cls.from_native(_col(column_name))

    @property
    def native(self) -> Expression:
        return self._expr

    def not_(self) -> Self:
        return self.from_native(~self.native)

    def alias(self, alias: str, /) -> _Expr:
        return self.from_native(self.native.alias(alias))

    def is_between(self, lower: IntoExpr, upper: IntoExpr) -> _Expr:
        return self.from_native(
            self.native.between(_into_native_expr(lower), _into_native_expr(upper))
        )

    def asc(self) -> _Expr:
        return self.from_native(self.native.asc())

    def desc(self) -> _Expr:
        return self.from_native(self.native.desc())

    def nulls_first(self) -> _Expr:
        return self.from_native(self.native.nulls_first())

    def nulls_last(self) -> _Expr:
        return self.from_native(self.native.nulls_last())

    def is_null(self) -> _Expr:
        return self.from_native(self.native.isnull())

    def is_not_null(self) -> _Expr:
        return self.from_native(self.native.isnotnull())

    def is_in(self, *cols: IntoExpr) -> _Expr:
        return self.from_native(self.native.isin(*_into_native_expr_iter(cols)))

    def is_not_in(self, *cols: IntoExpr) -> _Expr:
        return self.is_in(*cols).not_()


def _into_expr(value: IntoExpr, /, *, str_as_lit: bool = False) -> _Expr:
    if isinstance(value, _Expr):
        return value
    if _is_native_expr(value):
        return _Expr.from_native(value)
    if isinstance(value, str) and not str_as_lit:
        return _Expr.from_name(value)
    return lit(value)


def _into_native_expr(value: IntoExpr, /, *, str_as_lit: bool = False) -> Expression:
    if _is_native_expr(value):
        return value
    return _into_expr(value, str_as_lit=str_as_lit).native


def _flatten_once(into_exprs: IntoExpr | Iterable[IntoExpr], /) -> Iterator[IntoExpr]:
    if isinstance(into_exprs, Iterable) and not isinstance(into_exprs, (str, bytes)):
        yield from into_exprs
    else:
        yield into_exprs


def _into_native_expr_iter(
    into_exprs: IntoExpr | Iterable[IntoExpr],
    /,
    *more_exprs: IntoExpr,
    str_as_lit: bool = False,
) -> Iterator[Expression]:
    values: Iterable[IntoExpr]
    if not more_exprs:
        values = _flatten_once(into_exprs)
    else:
        values = chain(_flatten_once(into_exprs), more_exprs)
    for expr in values:
        yield _into_native_expr(expr, str_as_lit=str_as_lit)


def col(name: str, /) -> _Expr:
    return _Expr.from_name(name)


def lit(value: NonNestedLiteral, /) -> _Expr:
    return _Expr.from_scalar(value)
