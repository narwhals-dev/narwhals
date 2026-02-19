from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._function import Function, HorizontalFunction
from narwhals._plan._parse import parse_into_expr_ir
from narwhals._plan.expressions.namespace import ExprNamespace, IRNamespace
from narwhals._plan.options import FEOptions, FunctionOptions

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions import ExprIR, FunctionExpr as FExpr


# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
elementwise = FunctionOptions.elementwise
same_dtype = ResolveDType.function_same_dtype


# fmt: off
class StringFunction(Function, accessor="str", options=elementwise): ...
class LenChars(StringFunction, dtype=dtm.U32): ...
class ToLowercase(StringFunction, dtype=same_dtype()): ...
class ToUppercase(StringFunction, dtype=same_dtype()): ...
class ToTitlecase(StringFunction, dtype=same_dtype()): ...
class ConcatStr(HorizontalFunction, StringFunction, dtype=dtm.STR):
    __slots__ = ("ignore_nulls", "separator")
    separator: str
    ignore_nulls: bool
class Contains(StringFunction, dtype=dtm.BOOL):
    __slots__ = ("literal", "pattern")
    pattern: str
    literal: bool
class EndsWith(StringFunction, dtype=dtm.BOOL):
    __slots__ = ("suffix",)
    suffix: str
class Replace(StringFunction, dtype=same_dtype()):
    """N-ary (expr, value)."""

    __slots__ = ("literal", "n", "pattern")
    pattern: str
    literal: bool
    n: int
    def unwrap_input(self, node: FExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, value = node.input
        return expr, value

class ReplaceAll(StringFunction, dtype=same_dtype()):
    """N-ary (expr, value)."""

    __slots__ = ("literal", "pattern")
    pattern: str
    literal: bool
    def unwrap_input(self, node: FExpr[Self], /) -> tuple[ExprIR, ExprIR]:  # pragma: no cover
        expr, value = node.input
        return expr, value

    def to_replace_n(self, n: int) -> Replace:
        return Replace(pattern=self.pattern, literal=self.literal, n=n)

class Slice(StringFunction, dtype=same_dtype()):
    __slots__ = ("length", "offset")
    offset: int
    length: int | None
class Split(StringFunction, dtype=dtm.dtypes.List(dtm.STR)):
    __slots__ = ("by",)
    by: str
class StartsWith(StringFunction, dtype=dtm.BOOL):
    __slots__ = ("prefix",)
    prefix: str
class StripChars(StringFunction, dtype=same_dtype()):
    __slots__ = ("characters",)
    characters: str | None
class ToDate(StringFunction, dtype=dtm.DATE):
    __slots__ = ("format",)
    format: str | None
class ZFill(StringFunction, config=FEOptions.renamed("zfill"), dtype=same_dtype()):
    __slots__ = ("length",)
    length: int
# TODO @dangotbanned: Get @MarcoGorelli's opinion on `str.to_datetime` resolve_dtype.
# Can we work with (one of) `format: str | None`?
class ToDatetime(StringFunction):
    __slots__ = ("format",)
    format: str | None
# fmt: on


class IRStringNamespace(IRNamespace):
    len_chars: ClassVar = LenChars
    to_lowercase: ClassVar = ToLowercase
    to_uppercase: ClassVar = ToUppercase
    to_titlecase: ClassVar = ToTitlecase
    split: ClassVar = Split
    starts_with: ClassVar = StartsWith
    ends_with: ClassVar = EndsWith
    zfill: ClassVar = ZFill

    def replace(self, pattern: str, *, literal: bool = False, n: int = 1) -> Replace:
        return Replace(pattern=pattern, literal=literal, n=n)

    def replace_all(self, pattern: str, *, literal: bool = False) -> ReplaceAll:
        return ReplaceAll(pattern=pattern, literal=literal)

    def strip_chars(self, characters: str | None = None) -> StripChars:
        return StripChars(characters=characters)

    def contains(self, pattern: str, *, literal: bool = False) -> Contains:
        return Contains(pattern=pattern, literal=literal)

    def slice(self, offset: int, length: int | None = None) -> Slice:
        return Slice(offset=offset, length=length)

    def head(self, n: int = 5) -> Slice:
        return self.slice(0, n)

    def tail(self, n: int = 5) -> Slice:
        return self.slice(-n)

    def to_date(self, format: str | None = None) -> ToDate:  # pragma: no cover
        return ToDate(format=format)

    def to_datetime(self, format: str | None = None) -> ToDatetime:  # pragma: no cover
        return ToDatetime(format=format)


class ExprStringNamespace(ExprNamespace[IRStringNamespace]):
    @property
    def _ir_namespace(self) -> type[IRStringNamespace]:
        return IRStringNamespace

    def len_chars(self) -> Expr:
        return self._with_unary(self._ir.len_chars())

    def replace(
        self, pattern: str, value: str | Expr, *, literal: bool = False, n: int = 1
    ) -> Expr:
        other = parse_into_expr_ir(value, str_as_lit=True)
        replace = self._ir.replace(pattern, literal=literal, n=n)
        return self._expr._from_ir(replace.to_function_expr(self._expr._ir, other))

    def replace_all(
        self, pattern: str, value: str | Expr, *, literal: bool = False
    ) -> Expr:
        other = parse_into_expr_ir(value, str_as_lit=True)
        replace = self._ir.replace_all(pattern, literal=literal)
        return self._expr._from_ir(replace.to_function_expr(self._expr._ir, other))

    def strip_chars(self, characters: str | None = None) -> Expr:
        return self._with_unary(self._ir.strip_chars(characters))

    def starts_with(self, prefix: str) -> Expr:
        return self._with_unary(self._ir.starts_with(prefix=prefix))

    def ends_with(self, suffix: str) -> Expr:
        return self._with_unary(self._ir.ends_with(suffix=suffix))

    def contains(self, pattern: str, *, literal: bool = False) -> Expr:
        return self._with_unary(self._ir.contains(pattern, literal=literal))

    def slice(self, offset: int, length: int | None = None) -> Expr:
        return self._with_unary(self._ir.slice(offset, length))

    def head(self, n: int = 5) -> Expr:
        return self._with_unary(self._ir.head(n))

    def tail(self, n: int = 5) -> Expr:
        return self._with_unary(self._ir.tail(n))

    def split(self, by: str) -> Expr:
        return self._with_unary(self._ir.split(by=by))

    def to_date(self, format: str | None = None) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.to_date(format))

    def to_datetime(self, format: str | None = None) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.to_datetime(format))

    def to_lowercase(self) -> Expr:
        return self._with_unary(self._ir.to_lowercase())

    def to_uppercase(self) -> Expr:
        return self._with_unary(self._ir.to_uppercase())

    def to_titlecase(self) -> Expr:
        return self._with_unary(self._ir.to_titlecase())

    def zfill(self, length: int) -> Expr:
        return self._with_unary(self._ir.zfill(length=length))
