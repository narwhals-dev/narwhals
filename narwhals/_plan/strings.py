from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprNamespace, Function, IRNamespace
from narwhals._plan.options import FunctionFlags, FunctionOptions

if TYPE_CHECKING:
    from narwhals._plan.dummy import DummyExpr


class StringFunction(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return "StringFunction"


class ConcatHorizontal(StringFunction):
    """`nw.functions.concat_str`."""

    __slots__ = ("ignore_nulls", "separator")

    separator: str
    ignore_nulls: bool

    @property
    def function_options(self) -> FunctionOptions:
        return super().function_options.with_flags(FunctionFlags.INPUT_WILDCARD_EXPANSION)

    def __repr__(self) -> str:
        return "str.concat_horizontal"


class Contains(StringFunction):
    __slots__ = ("literal", "pattern")

    pattern: str
    literal: bool

    def __repr__(self) -> str:
        return "str.contains"


class EndsWith(StringFunction):
    __slots__ = ("suffix",)

    suffix: str

    def __repr__(self) -> str:
        return "str.ends_with"


class LenChars(StringFunction):
    def __repr__(self) -> str:
        return "str.len_chars"


class Replace(StringFunction):
    __slots__ = ("literal", "n", "pattern", "value")

    pattern: str
    value: str
    literal: bool
    n: int

    def __repr__(self) -> str:
        return "str.replace"


class ReplaceAll(StringFunction):
    """`polars` uses a single node for this and `Replace`.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L65-L70
    """

    __slots__ = ("literal", "pattern", "value")

    pattern: str
    value: str
    literal: bool

    def __repr__(self) -> str:
        return "str.replace_all"


class Slice(StringFunction):
    """We're using for `Head`, `Tail` as well.

    https://github.com/dangotbanned/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L87-L89

    I don't think it's likely we'll support `Expr` as inputs for this any time soon.
    """

    __slots__ = ("length", "offset")

    offset: int
    length: int | None

    def __repr__(self) -> str:
        return "str.slice"


class Split(StringFunction):
    __slots__ = ("by",)

    by: str

    def __repr__(self) -> str:
        return "str.split"


class StartsWith(StringFunction):
    __slots__ = ("prefix",)

    prefix: str

    def __repr__(self) -> str:
        return "str.starts_with"


class StripChars(StringFunction):
    __slots__ = ("characters",)

    characters: str | None

    def __repr__(self) -> str:
        return "str.strip_chars"


class ToDatetime(StringFunction):
    """`polars` uses `Strptime`.

    We've got a fairly different representation.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L112
    """

    __slots__ = ("format",)

    format: str | None

    def __repr__(self) -> str:
        return "str.to_datetime"


class ToLowercase(StringFunction):
    def __repr__(self) -> str:
        return "str.to_lowercase"


class ToUppercase(StringFunction):
    def __repr__(self) -> str:
        return "str.to_uppercase"


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

    def len_chars(self) -> DummyExpr:
        return self._to_narwhals(self._ir.len_chars().to_function_expr(self._expr._ir))

    def replace(
        self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> DummyExpr:
        return self._to_narwhals(
            self._ir.replace(pattern, value, literal=literal, n=n).to_function_expr(
                self._expr._ir
            )
        )

    def replace_all(
        self, pattern: str, value: str, *, literal: bool = False
    ) -> DummyExpr:
        return self._to_narwhals(
            self._ir.replace_all(pattern, value, literal=literal).to_function_expr(
                self._expr._ir
            )
        )

    def strip_chars(self, characters: str | None = None) -> DummyExpr:
        return self._to_narwhals(
            self._ir.strip_chars(characters).to_function_expr(self._expr._ir)
        )

    def starts_with(self, prefix: str) -> DummyExpr:
        return self._to_narwhals(
            self._ir.starts_with(prefix).to_function_expr(self._expr._ir)
        )

    def ends_with(self, suffix: str) -> DummyExpr:
        return self._to_narwhals(
            self._ir.ends_with(suffix).to_function_expr(self._expr._ir)
        )

    def contains(self, pattern: str, *, literal: bool = False) -> DummyExpr:
        return self._to_narwhals(
            self._ir.contains(pattern, literal=literal).to_function_expr(self._expr._ir)
        )

    def slice(self, offset: int, length: int | None = None) -> DummyExpr:
        return self._to_narwhals(
            self._ir.slice(offset, length).to_function_expr(self._expr._ir)
        )

    def head(self, n: int = 5) -> DummyExpr:
        return self._to_narwhals(self._ir.head(n).to_function_expr(self._expr._ir))

    def tail(self, n: int = 5) -> DummyExpr:
        return self._to_narwhals(self._ir.tail(n).to_function_expr(self._expr._ir))

    def split(self, by: str) -> DummyExpr:
        return self._to_narwhals(self._ir.split(by).to_function_expr(self._expr._ir))

    def to_datetime(self, format: str | None = None) -> DummyExpr:
        return self._to_narwhals(
            self._ir.to_datetime(format).to_function_expr(self._expr._ir)
        )

    def to_lowercase(self) -> DummyExpr:
        return self._to_narwhals(self._ir.to_lowercase().to_function_expr(self._expr._ir))

    def to_uppercase(self) -> DummyExpr:
        return self._to_narwhals(self._ir.to_uppercase().to_function_expr(self._expr._ir))
