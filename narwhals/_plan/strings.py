from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprNamespace, Function, IRNamespace
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from narwhals._plan.dummy import Expr


class StringFunction(Function, accessor="str", options=FunctionOptions.elementwise): ...


class ConcatHorizontal(StringFunction, options=FunctionOptions.horizontal):
    """`nw.functions.concat_str`."""

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
    """`polars` uses a single node for this and `Replace`.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L65-L70
    """

    __slots__ = ("literal", "pattern", "value")
    pattern: str
    value: str
    literal: bool


class Slice(StringFunction):
    """We're using for `Head`, `Tail` as well.

    https://github.com/dangotbanned/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L87-L89
    """

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
    """`polars` uses `Strptime`.

    We've got a fairly different representation.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L112
    """

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
        return self._to_narwhals(self._ir.len_chars().to_function_expr(self._expr._ir))

    def replace(
        self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> Expr:
        return self._to_narwhals(
            self._ir.replace(pattern, value, literal=literal, n=n).to_function_expr(
                self._expr._ir
            )
        )

    def replace_all(self, pattern: str, value: str, *, literal: bool = False) -> Expr:
        return self._to_narwhals(
            self._ir.replace_all(pattern, value, literal=literal).to_function_expr(
                self._expr._ir
            )
        )

    def strip_chars(self, characters: str | None = None) -> Expr:
        return self._to_narwhals(
            self._ir.strip_chars(characters).to_function_expr(self._expr._ir)
        )

    def starts_with(self, prefix: str) -> Expr:
        return self._to_narwhals(
            self._ir.starts_with(prefix).to_function_expr(self._expr._ir)
        )

    def ends_with(self, suffix: str) -> Expr:
        return self._to_narwhals(
            self._ir.ends_with(suffix).to_function_expr(self._expr._ir)
        )

    def contains(self, pattern: str, *, literal: bool = False) -> Expr:
        return self._to_narwhals(
            self._ir.contains(pattern, literal=literal).to_function_expr(self._expr._ir)
        )

    def slice(self, offset: int, length: int | None = None) -> Expr:
        return self._to_narwhals(
            self._ir.slice(offset, length).to_function_expr(self._expr._ir)
        )

    def head(self, n: int = 5) -> Expr:
        return self._to_narwhals(self._ir.head(n).to_function_expr(self._expr._ir))

    def tail(self, n: int = 5) -> Expr:
        return self._to_narwhals(self._ir.tail(n).to_function_expr(self._expr._ir))

    def split(self, by: str) -> Expr:
        return self._to_narwhals(self._ir.split(by).to_function_expr(self._expr._ir))

    def to_datetime(self, format: str | None = None) -> Expr:
        return self._to_narwhals(
            self._ir.to_datetime(format).to_function_expr(self._expr._ir)
        )

    def to_lowercase(self) -> Expr:
        return self._to_narwhals(self._ir.to_lowercase().to_function_expr(self._expr._ir))

    def to_uppercase(self) -> Expr:
        return self._to_narwhals(self._ir.to_uppercase().to_function_expr(self._expr._ir))
