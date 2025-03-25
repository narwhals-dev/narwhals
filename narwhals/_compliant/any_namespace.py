"""`Expr` and `Series` namespace accessor protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

from narwhals.utils import CompliantT_co
from narwhals.utils import _StoresCompliant

if TYPE_CHECKING:
    from typing import Callable

    from narwhals.typing import TimeUnit

__all__ = [
    "CatNamespace",
    "DateTimeNamespace",
    "ListNamespace",
    "NameNamespace",
    "StringNamespace",
    "StructNamespace",
]


class CatNamespace(_StoresCompliant[CompliantT_co], Protocol[CompliantT_co]):
    def get_categories(self) -> CompliantT_co: ...


class DateTimeNamespace(_StoresCompliant[CompliantT_co], Protocol[CompliantT_co]):
    def to_string(self, format: str) -> CompliantT_co: ...
    def replace_time_zone(self, time_zone: str | None) -> CompliantT_co: ...
    def convert_time_zone(self, time_zone: str) -> CompliantT_co: ...
    def timestamp(self, time_unit: TimeUnit) -> CompliantT_co: ...
    def date(self) -> CompliantT_co: ...
    def year(self) -> CompliantT_co: ...
    def month(self) -> CompliantT_co: ...
    def day(self) -> CompliantT_co: ...
    def hour(self) -> CompliantT_co: ...
    def minute(self) -> CompliantT_co: ...
    def second(self) -> CompliantT_co: ...
    def millisecond(self) -> CompliantT_co: ...
    def microsecond(self) -> CompliantT_co: ...
    def nanosecond(self) -> CompliantT_co: ...
    def ordinal_day(self) -> CompliantT_co: ...
    def weekday(self) -> CompliantT_co: ...
    def total_minutes(self) -> CompliantT_co: ...
    def total_seconds(self) -> CompliantT_co: ...
    def total_milliseconds(self) -> CompliantT_co: ...
    def total_microseconds(self) -> CompliantT_co: ...
    def total_nanoseconds(self) -> CompliantT_co: ...


class ListNamespace(_StoresCompliant[CompliantT_co], Protocol[CompliantT_co]):
    def len(self) -> CompliantT_co: ...


class NameNamespace(_StoresCompliant[CompliantT_co], Protocol[CompliantT_co]):
    def keep(self) -> CompliantT_co: ...
    def map(self, function: Callable[[str], str]) -> CompliantT_co: ...
    def prefix(self, prefix: str) -> CompliantT_co: ...
    def suffix(self, suffix: str) -> CompliantT_co: ...
    def to_lowercase(self) -> CompliantT_co: ...
    def to_uppercase(self) -> CompliantT_co: ...


class StringNamespace(_StoresCompliant[CompliantT_co], Protocol[CompliantT_co]):
    def len_chars(self) -> CompliantT_co: ...
    def replace(
        self, pattern: str, value: str, *, literal: bool, n: int
    ) -> CompliantT_co: ...
    def replace_all(
        self, pattern: str, value: str, *, literal: bool
    ) -> CompliantT_co: ...
    def strip_chars(self, characters: str | None) -> CompliantT_co: ...
    def starts_with(self, prefix: str) -> CompliantT_co: ...
    def ends_with(self, suffix: str) -> CompliantT_co: ...
    def contains(self, pattern: str, *, literal: bool) -> CompliantT_co: ...
    def slice(self, offset: int, length: int | None) -> CompliantT_co: ...
    def split(self, by: str) -> CompliantT_co: ...
    def to_datetime(self, format: str | None) -> CompliantT_co: ...
    def to_lowercase(self) -> CompliantT_co: ...
    def to_uppercase(self) -> CompliantT_co: ...


class StructNamespace(_StoresCompliant[CompliantT_co], Protocol[CompliantT_co]):
    def field(self, name: str) -> CompliantT_co: ...
