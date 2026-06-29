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
    from typing_extensions import Self

    from narwhals._typing import _LazyAllowedImpl
    from narwhals.dtypes import DType

NativeT = TypeVar("NativeT", NativeDuckDB, "ibis.Table")
Compliant: TypeAlias = CompliantLazyFrame[Any, NativeT, Any]


def _getattr_typing(self: LazyFrame[Any] | Series[Any], name: str) -> Any:
    if name in self._ALLOW:
        return getattr(self._compliant, name)
    raise unsupported_error(name)


class LazyFrame(Protocol[NativeT]):
    _compliant: Compliant[NativeT]
    _version: Version = Version.V1
    _implementation: ClassVar[_LazyAllowedImpl]
    _ALLOW: frozenset[str] = frozenset(
        (
            "native",
            "__native_namespace__",
            "_native_frame",
            "schema",
            "columns",
            "collect_schema",
        )
    )

    def __init__(self, compliant: Compliant[NativeT]) -> None:
        self._compliant = compliant

    def simple_select(self, *column_names: str) -> Self:
        return type(self)(self._compliant.simple_select(*column_names))

    def get_column(self, name: str) -> Series[NativeT]:
        return Series(self.simple_select(name))

    def __narwhals_dataframe__(self) -> Self:
        return self

    @classmethod
    def from_native(cls: type[LazyFrame[Any]], native: NativeT) -> LazyFrame[NativeT]:
        ns = Version.V1.namespace.from_native_object(native).compliant
        return cls(ns.from_native(native))

    def to_pandas(self) -> pd.DataFrame: ...
    def to_arrow(self) -> pa.Table: ...

    if TYPE_CHECKING:
        ...
    else:

        def __getattr__(self, name: str) -> Any:
            return _getattr_typing(self, name)


class Series(Generic[NativeT]):
    _lazy: LazyFrame[NativeT]
    _version: Version = Version.V1

    _ALLOW: frozenset[str] = frozenset(
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
            return _getattr_typing(self, name)
