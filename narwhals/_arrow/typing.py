from __future__ import annotations  # pragma: no cover

from typing import TYPE_CHECKING  # pragma: no cover
from typing import Any  # pragma: no cover
from typing import TypeVar  # pragma: no cover

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    import pyarrow as pa
    import pyarrow.compute as pc
    from pyarrow.__lib_pxi.types import (  # pyright: ignore[reportMissingModuleSource]
        _BasicDataType,  # noqa: F401
    )
    from pyarrow._stubs_typing import (  # pyright: ignore[reportMissingModuleSource]
        Indices,  # noqa: F401
    )
    from pyarrow._stubs_typing import (  # pyright: ignore[reportMissingModuleSource]
        Mask,  # noqa: F401
    )

    from narwhals._arrow.expr import ArrowExpr
    from narwhals._arrow.series import ArrowSeries

    IntoArrowExpr: TypeAlias = "ArrowExpr | ArrowSeries[Any]"

    StringScalar: TypeAlias = "pc.StringScalar"
    StringArray: TypeAlias = "pc.StringArray"
    StringArrayT = TypeVar("StringArrayT", bound=StringArray)
    DataTypeT_co = TypeVar("DataTypeT_co", bound="pa.DataType", covariant=True)

Incomplete: TypeAlias = Any  # pragma: no cover
"""
Marker for working code that fails on the stubs.

Common issues:
- Annotated for `Array`, but not `ChunkedArray`
- Relies on typing information that the stubs don't provide statically
- Missing attributes
- Incorrect return types
- Inconsistent use of generic/concrete types
- `_clone_signature` used on signatures that are not identical
"""


StringScalarT = TypeVar("StringScalarT", bound="StringScalar")  # pragma: no cover
