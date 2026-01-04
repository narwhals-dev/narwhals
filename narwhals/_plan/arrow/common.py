"""Behavior shared by two or more classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Generic

from narwhals._plan.arrow import compat
from narwhals._plan.arrow.functions import random_indices
from narwhals._plan.arrow.guards import is_series
from narwhals._typing_compat import TypeVar
from narwhals._utils import Implementation, Version, _StoresNative

if TYPE_CHECKING:
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._plan.arrow.namespace import ArrowNamespace
    from narwhals._plan.arrow.typing import ChunkedArrayAny, Indices


NativeT = TypeVar("NativeT", "pa.Table", "ChunkedArrayAny")


class ArrowFrameSeries(Generic[NativeT]):
    implementation: ClassVar = Implementation.PYARROW
    _native: NativeT
    _version: Version

    # NOTE: Aliases to integrate with `@requires.backend_version`
    _backend_version = compat.BACKEND_VERSION
    _implementation = implementation

    @property
    def native(self) -> NativeT:
        return self._native

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._plan.arrow.namespace import ArrowNamespace

        return ArrowNamespace(self._version)

    def _with_native(self, native: NativeT) -> Self:
        msg = f"{type(self).__name__}._with_native"
        raise NotImplementedError(msg)

    def __len__(self) -> int:
        msg = f"{type(self).__name__}.__len__"
        raise NotImplementedError(msg)

    if compat.TAKE_ACCEPTS_TUPLE:

        def _gather(self, indices: Indices) -> NativeT:
            return self.native.take(indices)
    else:

        def _gather(self, indices: Indices) -> NativeT:
            rows = list(indices) if isinstance(indices, tuple) else indices
            return self.native.take(rows)

    def gather(self, indices: Indices | _StoresNative[ChunkedArrayAny]) -> Self:
        ca = self._gather(indices.native if is_series(indices) else indices)
        return self._with_native(ca)

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return self._with_native(self.native[offset::n])

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self._with_native(self.native.slice(offset=offset, length=length))

    def sample_n(
        self, n: int = 1, *, with_replacement: bool = False, seed: int | None = None
    ) -> Self:
        mask = random_indices(len(self), n, with_replacement=with_replacement, seed=seed)
        return self.gather(mask)
