from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._plan.compliant.plugins import BuiltinPlugin
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa
    from typing_extensions import TypeAlias, TypeIs

    from narwhals._native import NativeArrow
    from narwhals._plan.arrow.classes import ArrowClasses

# NOTE: `Never` might be another option?
# try that out if `Any` causes *any* issues
Unsupported: TypeAlias = Any


# TODO @dangotbanned: Move to `_plan.arrow.__init__.py`
class ArrowPlugin(
    BuiltinPlugin["ArrowClasses", "pa.Table", Unsupported, "pa.ChunkedArray[Any]"]
):
    __slots__ = ()
    implementation = Implementation.PYARROW
    sys_modules_targets = ("pyarrow",)

    def is_native(self, obj: Any) -> TypeIs[NativeArrow]:
        import pyarrow as pa

        return isinstance(obj, (pa.Table, pa.ChunkedArray))

    def native_dataframe_classes(self) -> Iterator[type[pa.Table]]:
        import pyarrow as pa

        yield pa.Table

    def native_series_classes(self) -> Iterator[type[pa.ChunkedArray[Any]]]:
        import pyarrow as pa

        yield pa.ChunkedArray

    def native_lazyframe_classes(self) -> Iterator[type[Unsupported]]:
        yield from ()

    @property
    def __narwhals_classes__(self) -> ArrowClasses:
        from narwhals._plan.arrow.classes import ArrowClasses

        return ArrowClasses()


ArrowPlugin()
