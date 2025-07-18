from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from functools import lru_cache
from itertools import chain, repeat
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar, overload

from narwhals._plan.common import _IMMUTABLE_HASH_NAME, Immutable, NamedIR
from narwhals.dtypes import Unknown

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView, ValuesView

    from typing_extensions import TypeAlias

    from narwhals._plan.contexts import ExprContext
    from narwhals._plan.typing import Seq
    from narwhals.dtypes import DType


IntoFrozenSchema: TypeAlias = (
    "Mapping[str, DType] | Iterator[tuple[str, DType]] | FrozenSchema"
)
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

    def project(
        self, exprs: Seq[NamedIR], context: ExprContext
    ) -> tuple[Seq[NamedIR], FrozenSchema]:
        if context.is_select():
            return exprs, self._select(exprs)
        if context.is_with_columns():
            return self._with_columns(exprs)
        raise TypeError(context)

    def _select(self, exprs: Seq[NamedIR]) -> FrozenSchema:
        """Return a new schema, equivalent to performing `df.select(*exprs)`.

        Arguments:
            exprs: Expanded, unaliased expressions.

        Notes:
            - New columns all use the `Unknown` dtype
            - Any `cast` nodes are not reflected in the schema
        """
        names = (e.name for e in exprs)
        default = Unknown()
        return freeze_schema((name, self.get(name, default)) for name in names)

    def _with_columns(self, exprs: Seq[NamedIR]) -> tuple[Seq[NamedIR], FrozenSchema]:
        exprs_out = deque[NamedIR]()
        named: dict[str, NamedIR[Any]] = {e.name: e for e in exprs}
        items: IntoFrozenSchema
        for name in self:
            exprs_out.append(named.pop(name, NamedIR.from_name(name)))
        if named:
            items = chain(self.items(), zip(named, repeat(Unknown(), len(named))))
            exprs_out.extend(named.values())
        else:
            items = self
        return tuple(exprs_out), freeze_schema(items)

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
    iterable: IntoFrozenSchema | None = None, /, **schema: DType
) -> FrozenSchema:
    if isinstance(iterable, FrozenSchema):
        return iterable
    into = iterable or schema
    hashable = tuple(into.items() if isinstance(into, Mapping) else into)
    return _freeze_schema_cache(hashable)


@lru_cache(maxsize=100)
def _freeze_schema_cache(schema: _FrozenSchemaHash, /) -> FrozenSchema:
    return FrozenSchema._from_hash_safe(schema)


@lru_cache(maxsize=100)
def freeze_columns(schema: FrozenSchema, /) -> FrozenColumns:
    return tuple(schema)
