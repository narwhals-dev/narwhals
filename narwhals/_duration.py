"""Tools for working with the Polars duration string language."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

IntervalUnit: TypeAlias = Literal[
    "ns", "us", "ms", "mo", "m", "s", "h", "d", "w", "q", "y"
]

PATTERN_INTERVAL: re.Pattern[str] = re.compile(
    r"^(?P<multiple>\d+)(?P<unit>ns|us|ms|mo|m|s|h|d|w|q|y)$"
)


def parse_interval_string(every: str) -> tuple[int, IntervalUnit]:
    """Parse a string like "1d", "2h", "3m" into a tuple of (number, unit).

    Returns:
        A tuple of multiple and unit parsed from the interval string.
    """
    if match := PATTERN_INTERVAL.match(every):
        return int(match["multiple"]), cast("IntervalUnit", match["unit"])
    msg = f"Invalid `every` string: {every}."
    raise ValueError(msg)
