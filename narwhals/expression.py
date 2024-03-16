from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable

from narwhals.dtypes import translate_dtype
from narwhals.utils import flatten

if TYPE_CHECKING:
    from narwhals.typing import IntoExpr


def extract_native(expr: Expr, other: Any) -> Any:
    if isinstance(other, Expr):
        return other._call(expr)
    return other


class Expr:
    def __init__(self, call: Callable[[Any], Any]) -> None:
        # callable from namespace to expr
        self._call = call

    # --- convert ---
    def alias(self, name: str) -> Expr:
        return self.__class__(lambda plx: self._call(plx).alias(name))

    def cast(
        self,
        dtype: Any,
    ) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).cast(translate_dtype(plx, dtype)),
        )

    # --- binary ---
    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
        return self.__class__(
            lambda plx: self._call(plx).__eq__(extract_native(plx, other))
        )

    def __and__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__and__(extract_native(plx, other))
        )

    def __or__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__or__(extract_native(plx, other))
        )

    def __add__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__add__(extract_native(plx, other))
        )

    def __radd__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__radd__(extract_native(plx, other))
        )

    def __sub__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__sub__(extract_native(plx, other))
        )

    def __rsub__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__rsub__(extract_native(plx, other))
        )

    def __truediv__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__truediv__(extract_native(plx, other))
        )

    def __rtruediv__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__rtruediv__(extract_native(plx, other))
        )

    def __mul__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__mul__(extract_native(plx, other))
        )

    def __rmul__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__rmul__(extract_native(plx, other))
        )

    def __le__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__le__(extract_native(plx, other))
        )

    def __lt__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__lt__(extract_native(plx, other))
        )

    def __gt__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__gt__(extract_native(plx, other))
        )

    def __ge__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__ge__(extract_native(plx, other))
        )

    # --- unary ---
    def mean(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).mean())

    def sum(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).sum())

    def min(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).min())

    def max(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).max())

    def n_unique(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).n_unique())

    def unique(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).unique())

    # --- transform ---
    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).is_between(lower_bound, upper_bound, closed)
        )

    def is_in(self, other: Any) -> Expr:
        return self.__class__(lambda plx: self._call(plx).is_in(other))

    def is_null(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).is_null())

    # --- partial reduction ---
    def drop_nulls(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).drop_nulls())

    def sample(self, n: int, fraction: float, *, with_replacement: bool) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).sample(
                n, fraction=fraction, with_replacement=with_replacement
            )
        )


def col(*names: str | Iterable[str]) -> Expr:
    return Expr(lambda plx: plx.col(*names))


def all() -> Expr:
    return Expr(lambda plx: plx.all())


def len() -> Expr:
    return Expr(lambda plx: plx.len())


def sum(*columns: str) -> Expr:
    return Expr(lambda plx: plx.sum(*columns))


def mean(*columns: str) -> Expr:
    return Expr(lambda plx: plx.mean(*columns))


def min(*columns: str) -> Expr:
    return Expr(lambda plx: plx.min(*columns))


def max(*columns: str) -> Expr:
    return Expr(lambda plx: plx.max(*columns))


def sum_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    return Expr(
        lambda plx: plx.sum_horizontal([extract_native(plx, v) for v in flatten(exprs)])
    )


__all__ = [
    "Expr",
]
