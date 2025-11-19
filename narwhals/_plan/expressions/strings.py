from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from narwhals._plan._function import Function, HorizontalFunction
from narwhals._plan.expressions.namespace import ExprNamespace, IRNamespace
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr


# fmt: off
class StringFunction(Function, accessor="str", options=FunctionOptions.elementwise): ...
class LenChars(StringFunction): ...
class ToLowercase(StringFunction): ...
class ToUppercase(StringFunction): ...
# fmt: on
class ConcatStr(HorizontalFunction, StringFunction):
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


class IRStringNamespace(IRNamespace):
    len_chars: ClassVar = LenChars
    to_lowercase: ClassVar = ToUppercase
    to_uppercase: ClassVar = ToLowercase
    split: ClassVar = Split
    starts_with: ClassVar = StartsWith
    ends_with: ClassVar = EndsWith

    def replace(
        self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> Replace:  # pragma: no cover
        return Replace(pattern=pattern, value=value, literal=literal, n=n)

    def replace_all(
        self, pattern: str, value: str, *, literal: bool = False
    ) -> ReplaceAll:  # pragma: no cover
        return ReplaceAll(pattern=pattern, value=value, literal=literal)

    def strip_chars(
        self, characters: str | None = None
    ) -> StripChars:  # pragma: no cover
        return StripChars(characters=characters)

    def contains(self, pattern: str, *, literal: bool = False) -> Contains:
        return Contains(pattern=pattern, literal=literal)

    def slice(self, offset: int, length: int | None = None) -> Slice:  # pragma: no cover
        return Slice(offset=offset, length=length)

    def head(self, n: int = 5) -> Slice:  # pragma: no cover
        return self.slice(0, n)

    def tail(self, n: int = 5) -> Slice:  # pragma: no cover
        return self.slice(-n)

    def to_datetime(self, format: str | None = None) -> ToDatetime:  # pragma: no cover
        return ToDatetime(format=format)


class ExprStringNamespace(ExprNamespace[IRStringNamespace]):
    @property
    def _ir_namespace(self) -> type[IRStringNamespace]:
        return IRStringNamespace

    def len_chars(self) -> Expr:
        return self._with_unary(self._ir.len_chars())

    def replace(
        self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.replace(pattern, value, literal=literal, n=n))

    def replace_all(
        self, pattern: str, value: str, *, literal: bool = False
    ) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.replace_all(pattern, value, literal=literal))

    def strip_chars(self, characters: str | None = None) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.strip_chars(characters))

    def starts_with(self, prefix: str) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.starts_with(prefix=prefix))

    def ends_with(self, suffix: str) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.ends_with(suffix=suffix))

    def contains(self, pattern: str, *, literal: bool = False) -> Expr:
        return self._with_unary(self._ir.contains(pattern, literal=literal))

    def slice(self, offset: int, length: int | None = None) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.slice(offset, length))

    def head(self, n: int = 5) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.head(n))

    def tail(self, n: int = 5) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.tail(n))

    def split(self, by: str) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.split(by=by))

    def to_datetime(self, format: str | None = None) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.to_datetime(format))

    def to_lowercase(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.to_lowercase())

    def to_uppercase(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.to_uppercase())
