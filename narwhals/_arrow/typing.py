from __future__ import annotations  # pragma: no cover

from typing import TYPE_CHECKING  # pragma: no cover
from typing import Any  # pragma: no cover
from typing import TypeVar  # pragma: no cover

if TYPE_CHECKING:
    import sys
    from typing import Generic
    from typing import Literal

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    import pyarrow as pa
    import pyarrow.compute as pc
    from pyarrow._stubs_typing import (  # pyright: ignore[reportMissingModuleSource]
        Indices,  # noqa: F401
    )
    from pyarrow._stubs_typing import (  # pyright: ignore[reportMissingModuleSource]
        Mask,  # noqa: F401
    )
    from pyarrow._stubs_typing import (  # pyright: ignore[reportMissingModuleSource]
        Order,  # noqa: F401
    )

    from narwhals._arrow.expr import ArrowExpr
    from narwhals._arrow.series import ArrowSeries

    IntoArrowExpr: TypeAlias = "ArrowExpr | ArrowSeries[Any]"
    TieBreaker: TypeAlias = Literal["min", "max", "first", "dense"]
    NullPlacement: TypeAlias = Literal["at_start", "at_end"]

    StringScalar: TypeAlias = "pc.StringScalar"
    StringArray: TypeAlias = "pc.StringArray"
    StringArrayT = TypeVar("StringArrayT", bound=StringArray)
    DataTypeT_co = TypeVar("DataTypeT_co", bound="pa.DataType", covariant=True)
    _AsPyType = TypeVar("_AsPyType")

    class _BasicDataType(pa.DataType, Generic[_AsPyType]): ...


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

ArrowScalarT_co = TypeVar(
    "ArrowScalarT_co", bound="pa.Scalar", covariant=True
)  # pragma: no cover
StringScalarT = TypeVar("StringScalarT", bound="StringScalar")  # pragma: no cover
