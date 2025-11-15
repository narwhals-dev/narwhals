"""Behavior shared by two or more classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic

from narwhals._plan.arrow.functions import BACKEND_VERSION
from narwhals._typing_compat import TypeVar
from narwhals._utils import Implementation, Version, _StoresNative

if TYPE_CHECKING:
    import pyarrow as pa
    from typing_extensions import Self, TypeIs

    from narwhals._plan.arrow.namespace import ArrowNamespace
    from narwhals._plan.arrow.typing import ChunkedArrayAny, SizedMultiIndexSelector


def is_series(obj: Any) -> TypeIs[_StoresNative[ChunkedArrayAny]]:
    from narwhals._plan.arrow.series import ArrowSeries

    return isinstance(obj, ArrowSeries)


NativeT = TypeVar("NativeT", "pa.Table", "ChunkedArrayAny")


class ArrowFrameSeries(Generic[NativeT]):
    implementation: ClassVar = Implementation.PYARROW
    _native: NativeT
    _version: Version

    @property
    def native(self) -> NativeT:
        return self._native

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._plan.arrow.namespace import ArrowNamespace

        return ArrowNamespace(self._version)

    def _with_native(self, native: NativeT) -> Self:
        msg = f"{type(self).__name__}._with_native"
        raise NotImplementedError(msg)

    if BACKEND_VERSION >= (18,):

        def _gather(self, indices: SizedMultiIndexSelector) -> NativeT:
            return self.native.take(indices)
    else:

        def _gather(self, indices: SizedMultiIndexSelector) -> NativeT:
            rows = list(indices) if isinstance(indices, tuple) else indices
            return self.native.take(rows)

    def gather(
        self, indices: SizedMultiIndexSelector | _StoresNative[ChunkedArrayAny]
    ) -> Self:
        ca = self._gather(indices.native if is_series(indices) else indices)
        return self._with_native(ca)

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self._with_native(self.native.slice(offset=offset, length=length))
