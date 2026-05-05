from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from narwhals._plan.compliant.typing import (
    DataFrameT_co,
    ExprT_co,
    FrameT_contra as FrameT,
    SeriesT_co,
)

if TYPE_CHECKING:
    from narwhals._plan.expressions import (
        FunctionExpr as FExpr,
        lists,
        strings,
        temporal as dt,
    )
    from narwhals._plan.expressions.categorical import GetCategories
    from narwhals._plan.expressions.struct import FieldByName
    from narwhals._utils import Version
    from narwhals.schema import Schema


# TODO @dangotbanned: (After fixing expr) avoid `FrameT`
class ExprCatNamespace(Protocol[FrameT, ExprT_co]):
    """`[FrameT, ExprT_co]`."""

    __slots__ = ()

    def get_categories(
        self, node: FExpr[GetCategories], frame: FrameT, name: str, /
    ) -> ExprT_co: ...


# TODO @dangotbanned: (After fixing expr) avoid `FrameT`
class ExprDateTimeNamespace(Protocol[FrameT, ExprT_co]):
    """`[FrameT, ExprT_co]`."""

    __slots__ = ()

    def date(self, node: FExpr[dt.Date], frame: FrameT, name: str, /) -> ExprT_co: ...
    def year(self, node: FExpr[dt.Year], frame: FrameT, name: str, /) -> ExprT_co: ...
    def month(self, node: FExpr[dt.Month], frame: FrameT, name: str, /) -> ExprT_co: ...
    def day(self, node: FExpr[dt.Day], frame: FrameT, name: str, /) -> ExprT_co: ...
    def hour(self, node: FExpr[dt.Hour], frame: FrameT, name: str, /) -> ExprT_co: ...
    def minute(self, node: FExpr[dt.Minute], frame: FrameT, name: str, /) -> ExprT_co: ...
    def second(self, node: FExpr[dt.Second], frame: FrameT, name: str, /) -> ExprT_co: ...
    def millisecond(
        self, node: FExpr[dt.Millisecond], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def microsecond(
        self, node: FExpr[dt.Microsecond], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def nanosecond(
        self, node: FExpr[dt.Nanosecond], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def ordinal_day(
        self, node: FExpr[dt.OrdinalDay], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def weekday(
        self, node: FExpr[dt.WeekDay], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def total_minutes(
        self, node: FExpr[dt.TotalMinutes], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def total_seconds(
        self, node: FExpr[dt.TotalSeconds], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def total_milliseconds(
        self, node: FExpr[dt.TotalMilliseconds], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def total_microseconds(
        self, node: FExpr[dt.TotalMicroseconds], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def total_nanoseconds(
        self, node: FExpr[dt.TotalNanoseconds], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def to_string(
        self, node: FExpr[dt.ToString], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def replace_time_zone(
        self, node: FExpr[dt.ReplaceTimeZone], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def convert_time_zone(
        self, node: FExpr[dt.ConvertTimeZone], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def offset_by(
        self, node: FExpr[dt.OffsetBy], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def truncate(
        self, node: FExpr[dt.Truncate], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def timestamp(
        self, node: FExpr[dt.Timestamp], frame: FrameT, name: str, /
    ) -> ExprT_co: ...


# TODO @dangotbanned: (After fixing expr) avoid `FrameT`
class ExprListNamespace(Protocol[FrameT, ExprT_co]):
    """`[FrameT, ExprT_co]`."""

    __slots__ = ()

    def contains(
        self, node: FExpr[lists.Contains], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def get(self, node: FExpr[lists.Get], frame: FrameT, name: str, /) -> ExprT_co: ...
    def len(self, node: FExpr[lists.Len], frame: FrameT, name: str, /) -> ExprT_co: ...
    def unique(
        self, node: FExpr[lists.Unique], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def join(self, node: FExpr[lists.Join], frame: FrameT, name: str, /) -> ExprT_co: ...
    def min(self, node: FExpr[lists.Min], frame: FrameT, name: str, /) -> ExprT_co: ...
    def max(self, node: FExpr[lists.Max], frame: FrameT, name: str, /) -> ExprT_co: ...
    def mean(self, node: FExpr[lists.Mean], frame: FrameT, name: str, /) -> ExprT_co: ...
    def median(
        self, node: FExpr[lists.Median], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def sum(self, node: FExpr[lists.Sum], frame: FrameT, name: str, /) -> ExprT_co: ...
    def any(self, node: FExpr[lists.Any], frame: FrameT, name: str, /) -> ExprT_co: ...
    def all(self, node: FExpr[lists.All], frame: FrameT, name: str, /) -> ExprT_co: ...
    def first(
        self, node: FExpr[lists.First], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def last(self, node: FExpr[lists.Last], frame: FrameT, name: str, /) -> ExprT_co: ...
    def n_unique(
        self, node: FExpr[lists.NUnique], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def sort(self, node: FExpr[lists.Sort], frame: FrameT, name: str, /) -> ExprT_co: ...


# TODO @dangotbanned: (After fixing expr) avoid `FrameT`
class ExprStringNamespace(Protocol[FrameT, ExprT_co]):
    """`[FrameT, ExprT_co]`."""

    __slots__ = ()

    def contains(
        self, node: FExpr[strings.Contains], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def ends_with(
        self, node: FExpr[strings.EndsWith], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def len_chars(
        self, node: FExpr[strings.LenChars], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def replace(
        self, node: FExpr[strings.Replace], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def replace_all(
        self, node: FExpr[strings.ReplaceAll], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def slice(
        self, node: FExpr[strings.Slice], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def split(
        self, node: FExpr[strings.Split], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def starts_with(
        self, node: FExpr[strings.StartsWith], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def strip_chars(
        self, node: FExpr[strings.StripChars], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def to_uppercase(
        self, node: FExpr[strings.ToUppercase], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def to_lowercase(
        self, node: FExpr[strings.ToLowercase], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def to_titlecase(
        self, node: FExpr[strings.ToTitlecase], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def to_date(
        self, node: FExpr[strings.ToDate], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def to_datetime(
        self, node: FExpr[strings.ToDatetime], frame: FrameT, name: str, /
    ) -> ExprT_co: ...
    def zfill(
        self, node: FExpr[strings.ZFill], frame: FrameT, name: str, /
    ) -> ExprT_co: ...


# TODO @dangotbanned: (After fixing expr) avoid `FrameT`
class ExprStructNamespace(Protocol[FrameT, ExprT_co]):
    """`[FrameT, ExprT_co]`."""

    __slots__ = ()

    def field(
        self, node: FExpr[FieldByName], frame: FrameT, name: str, /
    ) -> ExprT_co: ...


class SeriesStructNamespace(Protocol[DataFrameT_co, SeriesT_co]):
    """`[DataFrameT_co, SeriesT_co]`."""

    __slots__ = ()

    def field(self, name: str) -> SeriesT_co: ...
    def unnest(self) -> DataFrameT_co: ...
    @property
    def schema(self) -> Schema: ...
    @property
    def compliant(self) -> SeriesT_co: ...
    @property
    def version(self) -> Version:
        return self.compliant.version
