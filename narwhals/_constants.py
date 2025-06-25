from __future__ import annotations

import datetime as dt

MAX_REPR_LENGTH = 40
MAX_WHILE_REPEAT = 100
NON_ELEMENTARY_DEPTH = 2

# Temporal (from `polars._utils.constants`)
SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
NS_PER_SECOND = 1_000_000_000
"""Nanoseconds per second."""
US_PER_SECOND = 1_000_000
"""Microseconds per second."""
MS_PER_SECOND = 1_000
"""Milliseconds per second."""

EPOCH_YEAR = 1970
EPOCH_DATE = dt.date(EPOCH_YEAR, 1, 1)
EPOCH = dt.datetime(EPOCH_YEAR, 1, 1).replace(tzinfo=None)
EPOCH_UTC = dt.datetime(EPOCH_YEAR, 1, 1, tzinfo=dt.timezone.utc)
