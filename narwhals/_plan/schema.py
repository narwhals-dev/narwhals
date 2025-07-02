from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeVar, overload

from narwhals._plan.common import _IMMUTABLE_HASH_NAME, Immutable

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView

    from typing_extensions import TypeAlias

    from narwhals._plan.typing import Seq
    from narwhals.dtypes import DType


IntoFrozenSchema: TypeAlias = "Mapping[str, DType] | FrozenSchema"
"""A schema to freeze, or an already frozen one.

As `DType` instances (`.values()`) are hashable, we can coerce the schema
into a cache-safe proxy structure (`FrozenSchema`).
"""

FrozenColumns: TypeAlias = "Seq[str]"
_FrozenSchemaHash: TypeAlias = "Seq[tuple[str, DType]]"
_T2 = TypeVar("_T2")


class FrozenSchema(Immutable):
    """Use `freeze_schema(...)` constructor to trigger caching!"""

    __slots__ = ("_mapping",)
    _mapping: MappingProxyType[str, DType]

    @property
    def __immutable_hash__(self) -> int:
        if hasattr(self, _IMMUTABLE_HASH_NAME):
            return self.__immutable_hash_value__
        hash_value = hash((self.__class__, *tuple(self._mapping.items())))
        object.__setattr__(self, _IMMUTABLE_HASH_NAME, hash_value)
        return self.__immutable_hash_value__

    @property
    def names(self) -> FrozenColumns:
        """Get the column names of the schema."""
        return freeze_columns(self)

    @staticmethod
    def _from_mapping(mapping: MappingProxyType[str, DType], /) -> FrozenSchema:
        return FrozenSchema(_mapping=mapping)

    @staticmethod
    def _from_hash_safe(items: _FrozenSchemaHash, /) -> FrozenSchema:
        clone = MappingProxyType(dict(items))
        return FrozenSchema._from_mapping(clone)

    def items(self) -> ItemsView[str, DType]:
        return self._mapping.items()

    def keys(self) -> KeysView[str]:
        return self._mapping.keys()

    def values(self) -> ValuesView[DType]:
        return self._mapping.values()

    @overload
    def get(self, key: str, /) -> DType | None: ...
    @overload
    def get(self, key: str, default: DType | _T2, /) -> DType | _T2: ...
    def get(self, key: str, default: DType | _T2 | None = None, /) -> DType | _T2 | None:
        if default is not None:
            return self._mapping.get(key, default)
        return self._mapping.get(key)

    def __iter__(self) -> Iterator[str]:
        yield from self._mapping

    def __contains__(self, key: object) -> bool:
        return self._mapping.__contains__(key)

    def __getitem__(self, key: str, /) -> DType:
        return self._mapping.__getitem__(key)

    def __len__(self) -> int:
        return self._mapping.__len__()

    def __repr__(self) -> str:
        sep, nl, indent = ",", "\n", " "
        items = f"{sep}{nl}{indent}".join(repr(tuple(els)) for els in self.items())
        return f"{type(self).__name__}([{nl}{indent}{items}{sep}{nl}])"


@overload
def freeze_schema(mapping: IntoFrozenSchema, /) -> FrozenSchema: ...
@overload
def freeze_schema(**schema: DType) -> FrozenSchema: ...
def freeze_schema(
    mapping: IntoFrozenSchema | None = None, /, **schema: DType
) -> FrozenSchema:
    if mapping and isinstance(mapping, FrozenSchema):
        return mapping
    schema_hash = tuple((mapping or schema).items())
    return _freeze_schema_cache(schema_hash)


@lru_cache(maxsize=100)
def _freeze_schema_cache(schema: _FrozenSchemaHash, /) -> FrozenSchema:
    return FrozenSchema._from_hash_safe(schema)


@lru_cache(maxsize=100)
def freeze_columns(schema: FrozenSchema, /) -> FrozenColumns:
    return tuple(schema)
