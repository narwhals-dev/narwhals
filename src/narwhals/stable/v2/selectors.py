from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals import selectors as nw_s

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import timezone

    from narwhals.dtypes import DType
    from narwhals.stable.v2 import Selector
    from narwhals.typing import TimeUnit


def _stableify(obj: nw_s.Selector, /) -> Selector:
    from narwhals.stable.v2 import Selector

    return Selector(*obj._nodes)


def all() -> Selector:
    """Select all columns."""
    return _stableify(nw_s.all())


def boolean() -> Selector:
    """Select boolean columns."""
    return _stableify(nw_s.boolean())


def by_dtype(*dtypes: DType | type[DType] | Iterable[DType | type[DType]]) -> Selector:
    """Select columns based on their dtype.

    Arguments:
        dtypes: one or data types to select
    """
    return _stableify(nw_s.by_dtype(*dtypes))


def categorical() -> Selector:
    """Select categorical columns."""
    return _stableify(nw_s.categorical())


def datetime(
    time_unit: TimeUnit | Iterable[TimeUnit] | None = None,
    time_zone: str | timezone | Iterable[str | timezone | None] | None = ("*", None),
) -> Selector:
    """Select all datetime columns, optionally filtering by time unit/zone.

    Arguments:
        time_unit: One (or more) of the allowed timeunit precision strings, "ms", "us",
            "ns" and "s". Omit to select columns with any valid timeunit.
        time_zone: Specify which timezone(s) to select

            * One or more timezone strings, as defined in zoneinfo (to see valid options
                run `import zoneinfo; zoneinfo.available_timezones()` for a full list).
            * Set `None` to select Datetime columns that do not have a timezone.
            * Set `"*"` to select Datetime columns that have *any* timezone.
    """
    return _stableify(nw_s.datetime(time_unit, time_zone))


def enum() -> Selector:
    """Select enum columns."""
    return _stableify(nw_s.enum())


def matches(pattern: str) -> Selector:
    """Select all columns that match the given regex pattern.

    Arguments:
        pattern: A valid regular expression pattern.
    """
    return _stableify(nw_s.matches(pattern))


def numeric() -> Selector:
    """Select numeric columns."""
    return _stableify(nw_s.numeric())


def string() -> Selector:
    """Select string columns."""
    return _stableify(nw_s.string())


__all__ = [
    "all",
    "boolean",
    "by_dtype",
    "categorical",
    "datetime",
    "enum",
    "matches",
    "numeric",
    "string",
]
