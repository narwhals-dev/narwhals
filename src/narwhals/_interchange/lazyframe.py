from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar

from narwhals._compliant import CompliantLazyFrame
from narwhals._interchange.dataframe import unsupported_error
from narwhals._native import NativeDuckDB
from narwhals._utils import Version

if TYPE_CHECKING:
    import ibis
    import pandas as pd
    import pyarrow as pa
    from typing_extensions import LiteralString, Self

    from narwhals._typing import _LazyAllowedImpl
    from narwhals.dtypes import DType

NativeT = TypeVar("NativeT", NativeDuckDB, "ibis.Table")
"""A native lazyframe."""
Compliant: TypeAlias = CompliantLazyFrame[Any, NativeT, Any]
"""An opaque compliant lazyframe wrapper."""
LazyFrameOps: TypeAlias = frozenset["LiteralString"]
"""Everything that is exposed from `CompliantLazyFrame` via `__getattr__`."""


class LazyFrame(Protocol[NativeT]):
    _compliant: Compliant[NativeT]
    _version: Version = Version.V1
    _implementation: ClassVar[_LazyAllowedImpl]
    _ALLOW: LazyFrameOps = frozenset(
        (
            "native",
            "__native_namespace__",
            "_native_frame",
            "schema",
            "columns",
            "collect_schema",
        )
    )

    def simple_select(self, *column_names: str) -> Self:
        return self.from_compliant(self._compliant.simple_select(*column_names))

    def get_column(self, name: str) -> Series[NativeT]:
        return Series(self.simple_select(name))

    def __narwhals_dataframe__(self) -> Self:
        return self

    @classmethod
    def from_compliant(cls, compliant: Compliant[NativeT]) -> Self:
        self = cls.__new__(cls)
        self._compliant = compliant
        return self

    @classmethod
    def from_native(cls, native: NativeT) -> Self:
        ns = Version.V1.namespace.from_native_object(native).compliant
        return cls.from_compliant(ns.from_native(native))

    def to_pandas(self) -> pd.DataFrame: ...
    def to_arrow(self) -> pa.Table: ...

    if TYPE_CHECKING:
        ...
    else:

        def __getattr__(self, name: str) -> Any:
            return _typed_getattr_impl(self, name)


class Series(Generic[NativeT]):
    _lazy: LazyFrame[NativeT]
    _version: Version = Version.V1
    _ALLOW: LazyFrameOps = frozenset(
        ("native", "__native_namespace__", "_implementation")
    )

    def __init__(self, lazy: LazyFrame[NativeT]) -> None:
        self._lazy = lazy

    def __narwhals_series__(self) -> Self:
        return self

    @property
    def _compliant(self) -> Compliant[NativeT]:
        return self._lazy._compliant

    @property
    def dtype(self) -> DType:
        return next(iter(self._compliant.schema.values()))

    if TYPE_CHECKING:
        ...
    else:

        def __getattr__(self, name: str) -> Any:
            return _typed_getattr_impl(self, name)


def _typed_getattr_impl(self: LazyFrame[Any] | Series[Any], name: str) -> Any:
    # NOTE: `self.__getattr__` is outside of type checking so that `self.i_dont_exist` is reported as an error
    # Splitting it out here ensures that features like "Find all references" still work
    if name in self._ALLOW:
        return getattr(self._compliant, name)
    raise unsupported_error(name)
