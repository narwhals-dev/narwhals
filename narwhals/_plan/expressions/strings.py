from __future__ import annotations

from typing import ClassVar

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._flags import FunctionFlags
from narwhals._plan._function import (
    BinaryFunction,
    Function,
    HorizontalFunction,
    UnaryFunction,
)
from narwhals._plan.expressions.namespace import IRNamespace

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
ELEMENTWISE = FunctionFlags.ELEMENTWISE
same_dtype = ResolveDType.function.same_dtype
renamed = DispatcherOptions.renamed


# fmt: off
class StringFunction(Function, dispatch=DispatcherOptions(accessor_name="str"), flags=ELEMENTWISE): ...
class _StringUnary(UnaryFunction, StringFunction): ...
class LenChars(_StringUnary, dtype=dtm.U32): ...
class ToLowercase(_StringUnary, dtype=same_dtype()): ...
class ToUppercase(_StringUnary, dtype=same_dtype()): ...
class ToTitlecase(_StringUnary, dtype=same_dtype()): ...
class ConcatStr(HorizontalFunction, StringFunction, dispatch=DispatcherOptions(), dtype=dtm.STR):
    __slots__ = ("ignore_nulls", "separator")
    separator: str
    ignore_nulls: bool
class Contains(_StringUnary, dtype=dtm.BOOL):
    __slots__ = ("literal", "pattern")
    pattern: str
    literal: bool
class EndsWith(_StringUnary, dtype=dtm.BOOL):
    __slots__ = ("suffix",)
    suffix: str
class Replace(BinaryFunction, StringFunction, dtype=same_dtype()):
    __slots__ = ("literal", "n", "pattern")
    pattern: str
    literal: bool
    n: int
class ReplaceAll(BinaryFunction, StringFunction, dtype=same_dtype()):
    __slots__ = ("literal", "pattern")
    pattern: str
    literal: bool
    def to_replace_n(self, n: int) -> Replace:
        return Replace(pattern=self.pattern, literal=self.literal, n=n)
class Slice(_StringUnary, dtype=same_dtype()):
    __slots__ = ("length", "offset")
    offset: int
    length: int | None
class Split(_StringUnary, dtype=dtm.dtypes.List(dtm.STR)):
    __slots__ = ("by",)
    by: str
class StartsWith(_StringUnary, dtype=dtm.BOOL):
    __slots__ = ("prefix",)
    prefix: str
class StripChars(_StringUnary, dtype=same_dtype()):
    __slots__ = ("characters",)
    characters: str | None
class ToDate(_StringUnary, dtype=dtm.DATE):
    __slots__ = ("format",)
    format: str | None
class ZFill(_StringUnary, dispatch=renamed("zfill"), dtype=same_dtype()):
    __slots__ = ("length",)
    length: int
# TODO @dangotbanned: Get @MarcoGorelli's opinion on `str.to_datetime` resolve_dtype.
# Can we work with (one of) `format: str | None`?
class ToDatetime(_StringUnary):
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
