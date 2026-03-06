"""Flags for features not available in all supported `polars` versions."""

from __future__ import annotations

from typing import Final

from narwhals._polars.utils import (
    SERIES_ACCEPTS_PD_INDEX as SERIES_ACCEPTS_PD_INDEX,  # noqa: PLC0414
    SERIES_RESPECTS_DTYPE as SERIES_RESPECTS_DTYPE,  # noqa: PLC0414
)
from narwhals._utils import Implementation

BACKEND_VERSION: Final = Implementation.POLARS._backend_version()
"""Static backend version for `polars`."""

# TODO @dangotbanned: Remove this dead code path on main, it is fixed at our minimum
HAS_POLARS_ERROR: Final = BACKEND_VERSION >= (0, 20, 4)
"""https://github.com/pola-rs/polars/pull/13615"""


SERIES_SORT_SUPPORTS_NULLS_LAST: Final = BACKEND_VERSION >= (0, 20, 6)
"""https://github.com/pola-rs/polars/pull/13794

Prior this this version, `nulls_last` was [only available on `*Frame` and `Expr`].

[only available on `*Frame` and `Expr`]: https://github.com/pola-rs/polars/issues/13788
"""

DUNDER_ARRAY_SUPPORTS_COPY: Final = BACKEND_VERSION >= (0, 20, 29)
"""https://github.com/pola-rs/polars/pull/16401"""

JOIN_OUTER_RENAMED_TO_FULL: Final = BACKEND_VERSION >= (0, 20, 29)
"""https://github.com/pola-rs/polars/pull/16417"""

MELT_RENAMED_TO_UNPIVOT: Final = BACKEND_VERSION >= (1, 0)
"""https://github.com/pola-rs/polars/pull/17095"""

CONSTRUCTOR_ACCEPTS_PYCAPSULE: Final = BACKEND_VERSION >= (1, 3)
"""https://github.com/pola-rs/polars/pull/17693"""

SERIES_HAS_FIRST_LAST: Final = BACKEND_VERSION >= (1, 10)
"""https://github.com/pola-rs/polars/pull/19093"""

SERIES_RFLOORDIV_HANDLES_ZERO: Final = BACKEND_VERSION >= (1, 10)
"""https://github.com/pola-rs/polars/issues/19142

Note:
    The bug impacts `__rmod__` as well, but didn't get fixed in narwhals?
"""

SERIES_RPOW_PRESERVES_NAME: Final = BACKEND_VERSION >= (1, 16, 1)
"""https://github.com/pola-rs/polars/pull/20072"""

IS_NAN_NUMERIC_PROPAGATES_NULLS: Final = BACKEND_VERSION >= (1, 18)
"""https://github.com/pola-rs/polars/pull/20386"""

MIN_PERIODS_RENAMED_TO_MIN_SAMPLES: Final = BACKEND_VERSION >= (1, 21)
"""https://github.com/pola-rs/polars/pull/20850"""

PIVOT_SUPPORTS_ON_COLUMNS: Final = BACKEND_VERSION >= (1, 36)
"""https://github.com/pola-rs/polars/pull/25016"""
