from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from functools import lru_cache
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeVar, final, overload

from narwhals._plan._expr_ir import NamedIR
from narwhals._plan._version import into_version
from narwhals._plan.exceptions import column_not_found_error
from narwhals._utils import (
    Version,
    _hasattr_static,
    check_column_names_are_unique,
    unstable,
)
from narwhals.dtypes import Unknown

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterable, Iterator, KeysView, ValuesView

    from typing_extensions import Never, Self, TypeAlias, TypeIs

    from narwhals._plan.typing import Seq
    from narwhals.dtypes import DType
    from narwhals.schema import Schema
    from narwhals.typing import IntoSchema


IntoFrozenSchema: TypeAlias = (
    "IntoSchema | Iterable[tuple[str, DType]] | FrozenSchema | HasSchema"
)
"""A schema to freeze, or an already frozen one.

As `DType` instances (`.values()`) are hashable, we can coerce the schema
into a cache-safe proxy structure (`FrozenSchema`).
"""

FrozenColumns: TypeAlias = "Seq[str]"
_FrozenSchemaHash: TypeAlias = "Seq[tuple[str, DType]]"
T = TypeVar("T")

_unknown = Unknown()
_OBJ_SETATTR = object.__setattr__
_HASH_NAME: Final = "_hash_value"


def set_flatten(iterable: Iterable[Iterator[T]], /) -> set[T]:
    """Equivalent to `set(chain.from_iterable(iterable))`.

    - Uses a set comprehension (despite the readability) since it has a unique [opcode].
    - *Technically* accepts `Iterable[Iterable[T]]`
      - but the typing overlaps with `str`

    [opcode]: https://docs.python.org/3.14/library/dis.html#opcode-SET_ADD
    """
    return {name for sub_it in iterable for name in sub_it}


@final
class FrozenSchema:
    """A cache-friendly - **internal** - extension to `nw.Schema`.

    - Not a subclass of `dict`, `Mapping`
      - But implements *enough* to look like it
    - Accepts **either** a single positional-only argument or **kwds
    - Has some extra methods inspired by [`polars-schema::schema::Schema`]

    Arguments:
        iterable: A schema, an iterable over one, or an object that defines `obj.schema: IntoSchema`.
        **schema: Keywords mapping column names to datatypes.

    Examples:
        >>> import narwhals as nw
        >>> mapping = {"a": nw.Int64(), "b": nw.String()}
        >>> schema = FrozenSchema(mapping)
        >>> schema
        FrozenSchema({'a': Int64, 'b': String})

        A `FrozenSchema` can be used *almost* anywhere a `Mapping` is expected:
        >>> nw.Schema(mapping) == nw.Schema(schema)
        True
        >>> mapping == dict(schema)
        True
        >>> dict(**schema)
        {'a': Int64, 'b': String}

        <!-- TODO @dangotbanned: Possibly fixable now that `Immutable` isn't involved
        - Runtime needs to do `Mapping.register(FrozenSchema)`
        - Type checking may work with some trickery using `object` as the real base class
        -->

        *Provided* the duck typing is respected:
        >>> from collections.abc import Mapping
        >>> isinstance(schema, (Mapping, dict))
        False

        Calls to `FrozenSchema` are cached:
        >>> schema is FrozenSchema(mapping)
        True

        While still providing the flexibilty of `dict()`:
        >>> schema is FrozenSchema(**mapping)
        True
        >>> schema is FrozenSchema(a=nw.Int64(), b=nw.String())
        True
        >>> schema == FrozenSchema(index=nw.UInt32(), **schema)
        False
        >>> schema is FrozenSchema(nw.Schema(mapping))
        True
        >>> schema is FrozenSchema(schema)
        True
        >>> schema is FrozenSchema(schema.items())
        True

        And then some:
        >>> from narwhals._plan import DataFrame
        >>> frame = DataFrame.from_dict({"a": [1], "b": ["ooh"]}, backend="polars")
        >>> schema is FrozenSchema(frame)
        True

    [`polars-schema::schema::Schema`]: https://github.com/pola-rs/polars/blob/5ee71f3ee4dd1573b45f44714da7843a6205895c/crates/polars-schema/src/schema.rs
    """

    __slots__ = (_HASH_NAME, "_mapping")
    # TODO @dangotbanned: (long-term) Add version branching for `frozendict` when available
    # https://docs.python.org/3.15/library/stdtypes.html#frozendict
    _mapping: MappingProxyType[str, DType]
    _hash_value: int

    # NOTE: Import, export
    @overload
    def __new__(cls, iterable: IntoFrozenSchema, /) -> FrozenSchema: ...
    # TODO @dangotbanned: (low-priority) Support the merging variant
    # https://docs.python.org/3.15/library/stdtypes.html#dict
    @overload
    def __new__(cls, /, **schema: DType) -> FrozenSchema: ...
    def __new__(
        cls, iterable: IntoFrozenSchema | None = None, /, **schema: DType
    ) -> FrozenSchema:
        # Making any kind of assignment in `__init__` would break the caching concept
        # https://docs.python.org/3/reference/datamodel.html#object.__new__
        if isinstance(iterable, FrozenSchema):
            return iterable
        into = iterable.schema if has_schema(iterable) else (iterable or schema)
        hashable = tuple(into.items() if isinstance(into, Mapping) else into)
        return _FrozenSchema_from_items_cached(hashable)

    @staticmethod
    def _from_mapping(mapping: MappingProxyType[str, DType], /) -> FrozenSchema:
        obj = object.__new__(FrozenSchema)
        _OBJ_SETATTR(obj, "_mapping", mapping)
        return obj

    @staticmethod
    def _from_items(items: _FrozenSchemaHash, /) -> FrozenSchema:
        # `frozendict` could slot in here, but should be a version-branch-defined function
        return FrozenSchema._from_mapping(MappingProxyType(dict(items)))

    def to_narwhals(self, version: Version = Version.MAIN) -> Schema:
        """Convert this `FrozenSchema` into `narwhals.Schema`."""
        return into_version(version).schema(self._mapping)

    # NOTE: Convenience methods/properties
    @property
    def names(self) -> FrozenColumns:
        """Get the column names of the schema."""
        return freeze_columns(self)

    def merge(self, other: FrozenSchema, /) -> FrozenSchema:
        """Return a new schema, merging `other` with `self` (see [upstream]).

        [upstream]: https://github.com/pola-rs/polars/blob/cdd247aaba8db3332be0bd031e0f31bc3fc33f77/crates/polars-schema/src/schema.rs#L265-L274.
        """
        return freeze_schema(self._mapping | other._mapping)

    # TODO @dangotbanned: Add resolved variant(s)?
    def select(self, exprs: Seq[NamedIR]) -> FrozenSchema:
        """Return a new schema, equivalent to performing `df.select(*exprs)`.

        Arguments:
            exprs: Expanded, unaliased expressions.

        Notes:
            - New columns all use the `Unknown` dtype
            - Any `cast` nodes are not reflected in the schema
        """
        names = (e.name for e in exprs)
        return freeze_schema((name, self.get(name, _unknown)) for name in names)

    def select_names(
        self,
        names: Collection[str],
        /,
        *,
        check_unique: bool = True,
        check_exists: bool = True,
    ) -> FrozenSchema:  # pragma: no cover
        """Return a new schema, equivalent to performing `df.select(names))`.

        Arguments:
            names: Column names to select.
            check_unique: Validate that `names` does not contain duplicates.
            check_exists: Validate that all `names` exist in the schema.
        """
        if check_unique or check_exists:
            requested = set(names)
            if check_unique and len(names) != len(requested):
                check_column_names_are_unique(names)
            if check_exists and not (self.keys() >= requested):
                raise column_not_found_error(names, self)
        return freeze_schema((name, self[name]) for name in names)

    def rename(
        self, mapping: Mapping[str, str], /, *, check_exists: bool = True
    ) -> FrozenSchema:  # pragma: no cover
        """Return a new schema, equivalent to performing `df.rename(mapping))`.

        Arguments:
            mapping: Key value pairs that map `{old: new}` names.
            check_exists: Validate that all keys (*old*) exist in the schema.
        """
        if not check_exists:
            it = ((mapping.get(name, name), dtype) for name, dtype in self.items())
            return freeze_schema(it)
        renames = mapping if isinstance(mapping, dict) else dict(mapping)
        old = tuple(renames)
        it = ((renames.pop(name, name), dtype) for name, dtype in self.items())
        result = freeze_schema(it)
        if renames:
            raise column_not_found_error(old, self)
        return result

    def with_columns(
        self,
        exprs: Seq[NamedIR],
        default: Callable[[str], DType | None] | Unknown = _unknown,
        /,
    ) -> FrozenSchema:  # pragma: no cover
        """Similar to `merge`, but preserving known `DType`s.

        When the incoming dtypes *at-least* partially known, `default` can be used to look them up:

            incoming: Mapping[str, DType] = {}

            partial = with_columns(exprs, incoming.get)
            full = with_columns(exprs, incoming.__getitem__)
        """
        names = (e.name for e in exprs)
        if not isinstance(default, Unknown):
            miss = {name: default(name) or _unknown for name in names if name not in self}
        else:
            miss = {name: _unknown for name in names if name not in self}
        return freeze_schema(self._mapping | miss)

    # TODO @dangotbanned: Update the other methods to try and avoid creating new schemas
    @unstable
    def with_columns_resolved(
        self, exprs: Seq[NamedIR], /
    ) -> FrozenSchema:  # pragma: no cover
        """Attempt to resolve the dtypes of each expression, merging each field into a new schema if needed."""
        current = self._mapping
        it = ((e.name, e.resolve_dtype(self)) for e in exprs)
        if updates := {
            name: dtype
            for name, dtype in it
            if name not in current or dtype != current[name]
        }:
            return freeze_schema(current | updates)
        return self

    def with_columns_irs(self, exprs: Seq[NamedIR]) -> Seq[NamedIR]:
        """Required for `concat(how="horizontal")`-based `with_columns`.

        Fills in any unreferenced columns present in `self`, but not in `exprs` as selections.
        """
        # NOTE: `mypy` is narrowing incorrectly on `dict.pop(..., None)`
        # https://github.com/python/mypy/issues/18297
        from narwhals._plan.expressions import col

        named = {e.name: e for e in exprs}
        it = (named.pop(name, None) or NamedIR(name, col(name)) for name in self)  # type: ignore[arg-type]
        return tuple(chain(it, named.values()))

    def contains_all(self, names: Iterable[Iterator[str]], /) -> bool:
        """Return True if this schema is a superset of columns in all of `names`."""
        return set_flatten(names).issubset(self._mapping)

    def __repr__(self) -> str:
        s = repr(dict(self))
        tp_name = type(self).__name__
        if (len(s) + len(tp_name) + 2) < 80:
            return f"{tp_name}({s})"
        # See https://github.com/astral-sh/ruff/discussions/22992
        s = "\n".join(f"{' ' * 4}{name!r}: {dtype}," for name, dtype in self.items())
        # NOTE: Use this after `1.19.*` https://github.com/python/mypy/pull/20325
        # `lb, rb = "{}"`
        lb, rb = "{", "}"
        hugging_parentheses = f"({lb}\n{s}\n{rb})"
        return f"{tp_name}{hugging_parentheses}"

    # NOTE: `Immutable`-related
    # `__str__`?
    def __hash__(self) -> int:
        if hasattr(self, _HASH_NAME):
            hash_value = self._hash_value
        else:
            hash_value = hash((FrozenSchema, *tuple(self.items())))
            _OBJ_SETATTR(self, _HASH_NAME, hash_value)
        return hash_value

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if type(other) is not FrozenSchema:
            return False
        return self._mapping.__eq__(other._mapping)

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: Any) -> Self:
        return self

    def __setattr__(self, name: str, value: Never) -> Never:
        if name not in self.__slots__:
            super().__setattr__(name, value)
        msg = f"{FrozenSchema.__name__!r} is immutable, {name!r} cannot be set."
        raise AttributeError(msg)

    if TYPE_CHECKING:
        ...
    else:

        def __delattr__(self, name: str) -> Never:
            if name not in self.__slots__:
                super().__delattr__(name)
            msg = f"{FrozenSchema.__name__!r} is immutable, {name!r} cannot be deleted."
            raise AttributeError(msg)

    def __init_subclass__(cls, **__: Never) -> Never:
        msg = f"Cannot subclass {FrozenSchema.__name__!r}"
        raise TypeError(msg)

    # NOTE: `Mapping` API
    @overload
    def get(self, key: str, /) -> DType | None: ...
    @overload
    def get(self, key: str, default: DType | T, /) -> DType | T: ...
    def get(self, key: str, default: DType | T | None = None, /) -> DType | T | None:
        if default is not None:
            return self._mapping.get(key, default)
        return self._mapping.get(key)

    def items(self) -> ItemsView[str, DType]:
        return self._mapping.items()

    def keys(self) -> KeysView[str]:
        return self._mapping.keys()

    def values(self) -> ValuesView[DType]:
        return self._mapping.values()

    def __contains__(self, key: object) -> bool:
        return self._mapping.__contains__(key)

    def __iter__(self) -> Iterator[str]:
        yield from self._mapping

    def __getitem__(self, key: str, /) -> DType:
        return self._mapping.__getitem__(key)

    def __len__(self) -> int:
        return self._mapping.__len__()


class HasSchema(Protocol):
    @property
    def schema(self) -> IntoSchema: ...


def has_schema(obj: Any) -> TypeIs[HasSchema]:
    return _hasattr_static(obj, "schema")


# TODO @dangotbanned: Update all the references to this **in the commit after**
freeze_schema = FrozenSchema


@lru_cache(maxsize=100)
def _FrozenSchema_from_items_cached(schema: _FrozenSchemaHash, /) -> FrozenSchema:  # noqa: N802
    return FrozenSchema._from_items(schema)


@lru_cache(maxsize=100)
def freeze_columns(schema: FrozenSchema, /) -> FrozenColumns:
    return tuple(schema)
