"""Tools for working with the Polars duration string language."""

from __future__ import annotations

import re

PATTERN_INTERVAL: re.Pattern[str] = re.compile(
    r"^(?P<multiple>\d+)(?P<unit>ns|us|ms|mo|m|s|h|d|w|q|y)$"
)


def parse_interval_string(every: str) -> tuple[int, str]:
    """Parse a string like "1d", "2h", "3m" into a tuple of (number, unit).

    Returns:
        A tuple of multiple and unit parsed from the interval string.
    """
    if match := PATTERN_INTERVAL.match(every):
        return int(match["multiple"]), match["unit"]
    msg = f"Invalid `every` string: {every}."
    raise ValueError(msg)
