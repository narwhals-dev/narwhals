from __future__ import annotations

import datetime as dt

# Temporal (from `polars._utils.constants`)
SECONDS_PER_DAY = 86_400
SECONDS_PER_MINUTE = 60
NS_PER_MINUTE = 60_000_000_000
"""Nanoseconds (`[ns]`) per minute."""
US_PER_MINUTE = 60_000_000
"""Microseconds (`[μs]`) per minute."""
MS_PER_MINUTE = 60_000
"""Milliseconds (`[ms]`) per minute."""
NS_PER_SECOND = 1_000_000_000
"""Nanoseconds (`[ns]`) per second (`[s]`)."""
US_PER_SECOND = 1_000_000
"""Microseconds (`[μs]`) per second (`[s]`)."""
MS_PER_SECOND = 1_000
"""Milliseconds (`[ms]`) per second (`[s]`)."""
NS_PER_MICROSECOND = 1_000
"""Nanoseconds (`[ns]`) per microsecond (`[μs]`)."""
NS_PER_MILLISECOND = 1_000_000
"""Nanoseconds (`[ns]`) per millisecond (`[ms]`).

From [polars](https://github.com/pola-rs/polars/blob/2c7a3e77f0faa37c86a3745db4ef7707ae50c72e/crates/polars-time/src/chunkedarray/duration.rs#L7).
"""
EPOCH_YEAR = 1970
"""See [Unix time](https://en.wikipedia.org/wiki/Unix_time)."""
EPOCH = dt.datetime(EPOCH_YEAR, 1, 1).replace(tzinfo=None)
"""See [Unix time](https://en.wikipedia.org/wiki/Unix_time)."""

# Some backends' default trim sets exclude whitespace recognized by Polars, such as
# tabs and newlines. Polars directly calls Rust's stdlib methods, which define
# whitespace according to the Unicode Derived Core Property White_Space.
# https://github.com/pola-rs/polars/blob/main/crates/polars-ops/src/chunked_array/strings/strip.rs
# https://doc.rust-lang.org/std/string/struct.String.html#method.trim_start
# https://www.unicode.org/Public/17.0.0/ucd/PropList.txt
POLARS_WHITESPACE = (
    "\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)
