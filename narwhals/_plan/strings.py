from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprNamespace, Function, IRNamespace
from narwhals._plan.options import FConfig, FunctionOptions

if TYPE_CHECKING:
    from narwhals._plan.dummy import Expr


class StringFunction(Function, accessor="str", options=FunctionOptions.elementwise): ...


class ConcatStr(
    StringFunction, options=FunctionOptions.horizontal, config=FConfig.namespaced()
):
    __slots__ = ("ignore_nulls", "separator")
    separator: str
    ignore_nulls: bool


class Contains(StringFunction):
    __slots__ = ("literal", "pattern")
    pattern: str
    literal: bool


class EndsWith(StringFunction):
    __slots__ = ("suffix",)
    suffix: str


class LenChars(StringFunction): ...


class Replace(StringFunction):
    __slots__ = ("literal", "n", "pattern", "value")
    pattern: str
    value: str
    literal: bool
    n: int


class ReplaceAll(StringFunction):
    __slots__ = ("literal", "pattern", "value")
    pattern: str
    value: str
    literal: bool


class Slice(StringFunction):
    __slots__ = ("length", "offset")
    offset: int
    length: int | None


class Split(StringFunction):
    __slots__ = ("by",)
    by: str


class StartsWith(StringFunction):
    __slots__ = ("prefix",)
    prefix: str


class StripChars(StringFunction):
    __slots__ = ("characters",)
    characters: str | None


class ToDatetime(StringFunction):
    __slots__ = ("format",)
    format: str | None


class ToLowercase(StringFunction): ...


class ToUppercase(StringFunction): ...


class IRStringNamespace(IRNamespace):
    def len_chars(self) -> LenChars:
        return LenChars()

    def replace(
        self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> Replace:
        return Replace(pattern=pattern, value=value, literal=literal, n=n)

    def replace_all(
        self, pattern: str, value: str, *, literal: bool = False
    ) -> ReplaceAll:
        return ReplaceAll(pattern=pattern, value=value, literal=literal)

    def strip_chars(self, characters: str | None = None) -> StripChars:
        return StripChars(characters=characters)

    def starts_with(self, prefix: str) -> StartsWith:
        return StartsWith(prefix=prefix)

    def ends_with(self, suffix: str) -> EndsWith:
        return EndsWith(suffix=suffix)

    def contains(self, pattern: str, *, literal: bool = False) -> Contains:
        return Contains(pattern=pattern, literal=literal)

    def slice(self, offset: int, length: int | None = None) -> Slice:
        return Slice(offset=offset, length=length)

    def head(self, n: int = 5) -> Slice:
        return self.slice(0, n)

    def tail(self, n: int = 5) -> Slice:
        return self.slice(-n)

    def split(self, by: str) -> Split:
        return Split(by=by)

    def to_datetime(self, format: str | None = None) -> ToDatetime:
        return ToDatetime(format=format)

    def to_lowercase(self) -> ToUppercase:
        return ToUppercase()

    def to_uppercase(self) -> ToLowercase:
        return ToLowercase()


class ExprStringNamespace(ExprNamespace[IRStringNamespace]):
    @property
    def _ir_namespace(self) -> type[IRStringNamespace]:
        return IRStringNamespace

    def len_chars(self) -> Expr:
        return self._with_unary(self._ir.len_chars())

    def replace(
        self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> Expr:
        return self._with_unary(self._ir.replace(pattern, value, literal=literal, n=n))

    def replace_all(self, pattern: str, value: str, *, literal: bool = False) -> Expr:
        return self._with_unary(self._ir.replace_all(pattern, value, literal=literal))

    def strip_chars(self, characters: str | None = None) -> Expr:
        return self._with_unary(self._ir.strip_chars(characters))

    def starts_with(self, prefix: str) -> Expr:
        return self._with_unary(self._ir.starts_with(prefix))

    def ends_with(self, suffix: str) -> Expr:
        return self._with_unary(self._ir.ends_with(suffix))

    def contains(self, pattern: str, *, literal: bool = False) -> Expr:
        return self._with_unary(self._ir.contains(pattern, literal=literal))

    def slice(self, offset: int, length: int | None = None) -> Expr:
        return self._with_unary(self._ir.slice(offset, length))

    def head(self, n: int = 5) -> Expr:
        return self._with_unary(self._ir.head(n))

    def tail(self, n: int = 5) -> Expr:
        return self._with_unary(self._ir.tail(n))

    def split(self, by: str) -> Expr:
        return self._with_unary(self._ir.split(by))

    def to_datetime(self, format: str | None = None) -> Expr:
        return self._with_unary(self._ir.to_datetime(format))

    def to_lowercase(self) -> Expr:
        return self._with_unary(self._ir.to_lowercase())

    def to_uppercase(self) -> Expr:
        return self._with_unary(self._ir.to_uppercase())
