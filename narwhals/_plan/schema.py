from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload

from narwhals._plan._expr_ir import NamedIR
from narwhals._plan._immutable import _IMMUTABLE_HASH_NAME, Immutable
from narwhals._utils import _hasattr_static
from narwhals.dtypes import Unknown

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView, ValuesView

    from typing_extensions import Never, TypeAlias, TypeIs

    from narwhals._plan.typing import Seq
    from narwhals.dtypes import DType
    from narwhals.typing import IntoSchema


IntoFrozenSchema: TypeAlias = (
    "IntoSchema | Iterator[tuple[str, DType]] | FrozenSchema | HasSchema"
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

    def __init_subclass__(cls, *_: Never, **__: Never) -> Never:
        msg = f"Cannot subclass {cls.__name__!r}"
        raise TypeError(msg)

    def merge(self, other: FrozenSchema, /) -> FrozenSchema:
        """Return a new schema, merging `other` with `self` (see [upstream]).

        [upstream]: https://github.com/pola-rs/polars/blob/cdd247aaba8db3332be0bd031e0f31bc3fc33f77/crates/polars-schema/src/schema.rs#L265-L274.
        """
        return freeze_schema(self._mapping | other._mapping)

    def select(self, exprs: Seq[NamedIR]) -> FrozenSchema:
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

    def select_irs(self, exprs: Seq[NamedIR]) -> Seq[NamedIR]:
        return exprs

    def with_columns(self, exprs: Seq[NamedIR]) -> FrozenSchema:
        # similar to `merge`, but preserving known `DType`s
        names = (e.name for e in exprs)
        default = Unknown()
        miss = {name: default for name in names if name not in self}
        return freeze_schema(self._mapping | miss)

    def with_columns_irs(self, exprs: Seq[NamedIR]) -> Seq[NamedIR]:
        """Required for `_concat_horizontal`-based `with_columns`.

        Fills in any unreferenced columns present in `self`, but not in `exprs` as selections.
        """
        named: dict[str, NamedIR[Any]] = {e.name: e for e in exprs}
        it = (named.pop(name, NamedIR.from_name(name)) for name in self)
        return tuple(chain(it, named.values()))

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
        obj = FrozenSchema.__new__(FrozenSchema)
        object.__setattr__(obj, "_mapping", mapping)
        return obj

    @staticmethod
    def _from_hash_safe(items: _FrozenSchemaHash, /) -> FrozenSchema:
        return FrozenSchema._from_mapping(MappingProxyType(dict(items)))

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


class HasSchema(Protocol):
    @property
    def schema(self) -> IntoSchema: ...


def has_schema(obj: Any) -> TypeIs[HasSchema]:
    return _hasattr_static(obj, "schema")


@overload
def freeze_schema(mapping: IntoFrozenSchema, /) -> FrozenSchema: ...
@overload
def freeze_schema(**schema: DType) -> FrozenSchema: ...
def freeze_schema(
    iterable: IntoFrozenSchema | None = None, /, **schema: DType
) -> FrozenSchema:
    if isinstance(iterable, FrozenSchema):
        return iterable
    into = iterable.schema if has_schema(iterable) else (iterable or schema)
    hashable = tuple(into.items() if isinstance(into, Mapping) else into)
    return _freeze_schema_cache(hashable)


@lru_cache(maxsize=100)
def _freeze_schema_cache(schema: _FrozenSchemaHash, /) -> FrozenSchema:
    return FrozenSchema._from_hash_safe(schema)


@lru_cache(maxsize=100)
def freeze_columns(schema: FrozenSchema, /) -> FrozenColumns:
    return tuple(schema)
