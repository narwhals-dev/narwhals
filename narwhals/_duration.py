"""Tools for working with the Polars duration string language."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, cast, get_args

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

__all__ = ["IntervalUnit", "parse_interval_string"]

IntervalUnit: TypeAlias = Literal["ns", "us", "ms", "s", "m", "h", "d", "mo", "q", "y"]
"""A Polars duration string interval unit.

- 'ns': nanosecond.
- 'us': microsecond.
- 'ms': millisecond.
- 's': second.
- 'm': minute.
- 'h': hour.
- 'd': day.
- 'mo': month.
- 'q': quarter.
- 'y': year.
"""

PATTERN_INTERVAL: re.Pattern[str] = re.compile(
    r"^(?P<multiple>\d+)(?P<unit>ns|us|ms|mo|m|s|h|d|q|y)\Z"
)
MONTH_MULTIPLES = frozenset([1, 2, 3, 4, 6, 12])
QUARTER_MULTIPLES = frozenset([1, 2, 4])


def parse_interval_string(every: str) -> tuple[int, IntervalUnit]:
    """Parse a string like "1d", "2h", "3m" into a tuple of (number, unit).

    Returns:
        A tuple of multiple and unit parsed from the interval string.
    """
    if match := PATTERN_INTERVAL.match(every):
        multiple = int(match["multiple"])
        unit = cast("IntervalUnit", match["unit"])
        if unit == "mo" and multiple not in MONTH_MULTIPLES:
            msg = f"Only the following multiples are supported for 'mo' unit: {MONTH_MULTIPLES}.\nGot: {multiple}."
            raise ValueError(msg)
        if unit == "q" and multiple not in QUARTER_MULTIPLES:
            msg = f"Only the following multiples are supported for 'q' unit: {QUARTER_MULTIPLES}.\nGot: {multiple}."
            raise ValueError(msg)
        if unit == "y" and multiple != 1:
            msg = (
                f"Only multiple 1 is currently supported for 'y' unit.\nGot: {multiple}."
            )
            raise ValueError(msg)
        return multiple, unit
    msg = (
        f"Invalid `every` string: {every}. Expected string of kind <number><unit>, "
        f"where 'unit' is one of: {get_args(IntervalUnit)}."
    )
    raise ValueError(msg)
