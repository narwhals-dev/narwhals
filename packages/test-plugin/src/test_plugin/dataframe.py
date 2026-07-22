from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._utils import (
    Implementation,
    ValidateBackendVersion,
    Version,
    not_implemented,
)
from narwhals.typing import CompliantLazyFrame

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import TypeAlias

    from typing_extensions import Self

    from narwhals import DataFrame, LazyFrame
    from narwhals._utils import _LimitedContext

DictFrame: TypeAlias = dict[str, list[Any]]


class DictDataFrame:
    """Minimal eager frame, kept to the smallest surface exercised in narwhals' tests."""

    _implementation = Implementation.UNKNOWN

    def __init__(self, native_dataframe: DictFrame, *, version: Version) -> None:
        self._native_frame: DictFrame = native_dataframe
        self._version = version

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        /,
        *,
        context: _LimitedContext,
        schema: Any = None,  # noqa: ARG003
    ) -> Self:
        return cls(
            {name: list(values) for name, values in data.items()},
            version=context._version,
        )

    @classmethod
    def from_dicts(
        cls,
        data: Sequence[Mapping[str, Any]],
        /,
        *,
        context: _LimitedContext,
        schema: Any = None,  # noqa: ARG003
    ) -> Self:
        columns: list[str] = list(data[0]) if data else []
        return cls(
            {name: [row[name] for row in data] for name in columns},
            version=context._version,
        )

    @classmethod
    def from_numpy(
        cls, data: Any, /, *, context: _LimitedContext, schema: Any = None
    ) -> Self:
        names = (
            list(schema)
            if schema is not None
            else [str(index) for index in range(data.shape[1])]
        )
        return cls(
            {name: data[:, index].tolist() for index, name in enumerate(names)},
            version=context._version,
        )

    @classmethod
    def from_arrow(cls, data: Any, /, *, context: _LimitedContext) -> Self:
        import pyarrow as pa

        return cls(pa.table(data).to_pydict(), version=context._version)

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> Any:
        from test_plugin.namespace import DictNamespace

        return DictNamespace(version=self._version)

    @property
    def native(self) -> DictFrame:
        return self._native_frame

    def to_narwhals(self) -> DataFrame[Any]:
        return self._version.dataframe(self, level="full")


class DictLazyFrame(
    CompliantLazyFrame[Any, "DictFrame", "LazyFrame[DictFrame]"],  # type: ignore[type-var]
    ValidateBackendVersion,
):
    _implementation = Implementation.UNKNOWN

    def __init__(self, native_dataframe: DictFrame, *, version: Version) -> None:
        self._native_frame: DictFrame = native_dataframe
        self._version = version

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def to_narwhals(self) -> LazyFrame[Any]:
        return self._version.lazyframe(self, level="lazy")

    @property
    def columns(self) -> list[str]:  # pragma: no cover
        return list(self._native_frame.keys())

    _with_native = not_implemented()

    def _with_version(self, version: Version) -> Self:
        return self.__class__(self._native_frame, version=version)

    # Dunders
    __narwhals_namespace__ = not_implemented()
    __native_namespace__ = not_implemented()

    # Properties
    schema: Any = not_implemented()

    # Static
    _is_native = not_implemented()

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
    tail = not_implemented()
    unique = not_implemented()
    unpivot = not_implemented()
    with_columns = not_implemented()
    with_row_index = not_implemented()
