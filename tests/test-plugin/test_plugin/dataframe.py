from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import daft
import daft.exceptions
import daft.functions

from narwhals._utils import (
    Implementation,
    ValidateBackendVersion,
    Version,
    not_implemented,
)
from narwhals.typing import CompliantLazyFrame

DictFrame: TypeAlias = dict[str, list[Any]]

if TYPE_CHECKING:
    from typing_extensions import Self, TypeIs

    from narwhals._utils import _LimitedContext
    from narwhals.dataframe import LazyFrame
    from narwhals.dtypes import DType


class DictLazyFrame(
    CompliantLazyFrame[Any, "DictFrame", "LazyFrame[DictFrame]"], ValidateBackendVersion
):
    _implementation = Implementation.UNKNOWN

    def __init__(self, native_dataframe: DictFrame, *, version: Version) -> None:
        self._native_frame: DictFrame = native_dataframe
        self._version = version
        self._cached_schema: dict[str, DType] | None = None
        self._cached_columns: list[str] | None = None

    @staticmethod
    def _is_native(obj: daft.DataFrame | Any) -> TypeIs[daft.DataFrame]:
        return isinstance(obj, daft.DataFrame)

    @classmethod
    def from_native(cls, data: daft.DataFrame, /, *, context: _LimitedContext) -> Self:
        return cls(data, version=context._version)

    def to_narwhals(self) -> LazyFrame[daft.DataFrame]:
        return self._version.lazyframe(self, level="lazy")

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def _with_version(self, version: Version) -> Self:
        return self.__class__(self._native_frame, version=version)

    @property
    def columns(self) -> list[str]:
        return list(self._native_frame.keys())

    group_by = not_implemented()
    join_asof = not_implemented()
    explode = not_implemented()
    sink_parquet = not_implemented()
