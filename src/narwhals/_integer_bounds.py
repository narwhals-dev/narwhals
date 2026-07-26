"""Min/max representable values per integer dtype, shared across backends.

Used by non-strict (`strict=False`) casting to null-out (rather than silently wrap
or raise on) out-of-range values before handing off to a backend's native cast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._utils import Version

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals.dtypes import DType

__all__ = ["INTEGER_BOUNDS"]

_dtypes = Version.MAIN.dtypes

INTEGER_BOUNDS: Mapping[type[DType], tuple[int, int]] = {
    _dtypes.Int8: (-128, 127),
    _dtypes.Int16: (-32_768, 32_767),
    _dtypes.Int32: (-2_147_483_648, 2_147_483_647),
    _dtypes.Int64: (-(2**63), 2**63 - 1),
    _dtypes.UInt8: (0, 255),
    _dtypes.UInt16: (0, 65_535),
    _dtypes.UInt32: (0, 2**32 - 1),
    _dtypes.UInt64: (0, 2**64 - 1),
}
