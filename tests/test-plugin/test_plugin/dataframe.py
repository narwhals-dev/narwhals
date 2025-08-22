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

    from narwhals.dataframe import LazyFrame


class DictLazyFrame(
    CompliantLazyFrame[Any, "DictFrame", "LazyFrame[DictFrame]"],
    ValidateBackendVersion,  # pyright: ignore[reportInvalidTypeArguments]
):
    _implementation = Implementation.UNKNOWN

    def __init__(self, native_dataframe: DictFrame, *, version: Version) -> None:
        self._native_frame: DictFrame = native_dataframe
        self._version = version

    @staticmethod
    def _is_native(obj: daft.DataFrame | Any) -> TypeIs[daft.DataFrame]:
        return isinstance(obj, daft.DataFrame)

    def to_narwhals(self) -> LazyFrame[daft.DataFrame]:
        return self._version.lazyframe(self, level="lazy")

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def _with_version(self, version: Version) -> Self:
        return self.__class__(self._native_frame, version=version)

    @property
    def columns(self) -> list[str]:
        return list(self._native_frame.keys())

    # Dunders
    __narwhals_namespace__ = not_implemented()

    # Properties
    schema = not_implemented()  # type: ignore[assignment]

    # Helpers
    _iter_columns = not_implemented()

    # Functions
    aggregate = not_implemented()
    collect = not_implemented()
    collect_schema = not_implemented()
    drop = not_implemented()
    drop_nulls = not_implemented()
    explode = not_implemented()
    filter = not_implemented()
    from_native = not_implemented()
    group_by = not_implemented()
    head = not_implemented()
    join = not_implemented()
    join_asof = not_implemented()
    rename = not_implemented()
    select = not_implemented()
    simple_select = not_implemented()
    sink_parquet = not_implemented()
    sort = not_implemented()
    unique = not_implemented()
    unpivot = not_implemented()
    with_columns = not_implemented()
    with_row_index = not_implemented()
