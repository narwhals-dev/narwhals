from __future__ import annotations

import datetime as dt

MAX_REPR_LENGTH = 40
MAX_WHILE_REPEAT = 100
NON_ELEMENTARY_DEPTH = 2

# Temporal (from `polars._utils.constants`)
SECONDS_PER_DAY = 86_400
SECONDS_PER_MINUTE = 60
NS_PER_MINUTE = 60_000_000_000
US_PER_MINUTE = 60_000_000
MS_PER_MINUTE = 60_000

NS_PER_SECOND = 1_000_000_000
"""Nanoseconds per second."""
US_PER_SECOND = 1_000_000
"""Microseconds per second."""
MS_PER_SECOND = 1_000
"""Milliseconds per second."""

NS_PER_MICROSECOND = 1_000
NS_PER_MILLISECOND = 1_000_000
"""https://github.com/pola-rs/polars/blob/2c7a3e77f0faa37c86a3745db4ef7707ae50c72e/crates/polars-time/src/chunkedarray/duration.rs#L7-L8"""

EPOCH_YEAR = 1970
EPOCH = dt.datetime(EPOCH_YEAR, 1, 1).replace(tzinfo=None)
