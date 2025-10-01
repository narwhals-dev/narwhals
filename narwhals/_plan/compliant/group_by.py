from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan._expansion import prepare_projection
from narwhals._plan._parse import parse_into_seq_of_expr_ir
from narwhals._plan.common import replace, temp
from narwhals._plan.compliant.typing import (
    DataFrameT,
    EagerDataFrameT,
    FrameT_co,
    ResolverT_co,
)
from narwhals.exceptions import ComputeError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

    from narwhals._plan.expressions import ExprIR, NamedIR
    from narwhals._plan.schema import FrozenSchema, IntoFrozenSchema
    from narwhals._plan.typing import IntoExpr, OneOrIterable, Seq


class CompliantGroupBy(Protocol[FrameT_co]):
    @property
    def compliant(self) -> FrameT_co: ...
    def agg(self, irs: Seq[NamedIR]) -> FrameT_co: ...


class DataFrameGroupBy(CompliantGroupBy[DataFrameT], Protocol[DataFrameT]):
    _keys: Seq[NamedIR]
    _key_names: Seq[str]

    @classmethod
    def from_resolver(
        cls, df: DataFrameT, resolver: GroupByResolver, /
    ) -> DataFrameGroupBy[DataFrameT]: ...
    @classmethod
    def by_names(
        cls, df: DataFrameT, names: Seq[str], /
    ) -> DataFrameGroupBy[DataFrameT]: ...
    def __iter__(self) -> Iterator[tuple[Any, DataFrameT]]: ...
    @property
    def keys(self) -> Seq[NamedIR]:
        return self._keys

    @property
    def key_names(self) -> Seq[str]:
        if names := self._key_names:
            return names
        msg = "at least one key is required in a group_by operation"
        raise ComputeError(msg)


class EagerDataFrameGroupBy(DataFrameGroupBy[EagerDataFrameT], Protocol[EagerDataFrameT]):
    _df: EagerDataFrameT
    _key_names: Seq[str]
    _key_names_original: Seq[str]
    _column_names_original: Seq[str]

    @classmethod
    def by_names(cls, df: EagerDataFrameT, names: Seq[str], /) -> Self:
        obj = cls.__new__(cls)
        obj._df = df
        obj._keys = ()
        obj._key_names = names
        obj._key_names_original = ()
        obj._column_names_original = tuple(df.columns)
        return obj

    @classmethod
    def from_resolver(
        cls, df: EagerDataFrameT, resolver: GroupByResolver, /
    ) -> EagerDataFrameGroupBy[EagerDataFrameT]:
        key_names = resolver.key_names
        if not resolver.requires_projection():
            df = df.drop_nulls(key_names) if resolver._drop_null_keys else df
            return cls.by_names(df, key_names)
        obj = cls.__new__(cls)
        unique_names = temp.column_names(chain(key_names, df.columns))
        safe_keys = tuple(
            replace(key, name=name) for key, name in zip(resolver.keys, unique_names)
        )
        obj._df = df.with_columns(resolver._schema_in.with_columns_irs(safe_keys))
        obj._keys = safe_keys
        obj._key_names = tuple(e.name for e in safe_keys)
        obj._key_names_original = key_names
        obj._column_names_original = resolver._schema_in.names
        return obj


class Grouper(Protocol[ResolverT_co]):
    """`GroupBy` helper for collecting and forwarding `Expr`s for projection.

    - Uses `Expr` everywhere (no need to duplicate layers)
    - Resolver only needs schema (neither needs a frame, but can use one to get `schema`)
    """

    _keys: Seq[ExprIR]
    _aggs: Seq[ExprIR]
    _drop_null_keys: bool

    @classmethod
    def by(cls, *by: OneOrIterable[IntoExpr]) -> Self:
        obj = cls.__new__(cls)
        obj._keys = parse_into_seq_of_expr_ir(*by)
        return obj

    def agg(self, *aggs: OneOrIterable[IntoExpr]) -> Self:
        self._aggs = parse_into_seq_of_expr_ir(*aggs)
        return self

    @property
    def _resolver(self) -> type[ResolverT_co]: ...

    def resolve(self, context: IntoFrozenSchema, /) -> ResolverT_co:
        """Project keys and aggs in `context`, expanding all `Expr` -> `NamedIR`."""
        return self._resolver.from_grouper(self, context)


class GroupByResolver:
    """Narwhals-level `GroupBy` resolver."""

    _schema_in: FrozenSchema
    _keys: Seq[NamedIR]
    _aggs: Seq[NamedIR]
    _key_names: Seq[str]
    _schema: FrozenSchema
    _drop_null_keys: bool

    @property
    def keys(self) -> Seq[NamedIR]:
        return self._keys

    @property
    def aggs(self) -> Seq[NamedIR]:
        return self._aggs

    @property
    def key_names(self) -> Seq[str]:
        if names := self._key_names:
            return names
        if keys := self.keys:
            return tuple(e.name for e in keys)
        msg = "at least one key is required in a group_by operation"
        raise ComputeError(msg)

    @property
    def schema(self) -> FrozenSchema:
        return self._schema

    def evaluate(self, frame: DataFrameT) -> DataFrameT:
        """Perform the `group_by` on `frame`."""
        return frame.group_by_resolver(self).agg(self.aggs)

    @classmethod
    def from_grouper(cls, grouper: Grouper[Self], context: IntoFrozenSchema, /) -> Self:
        """Loosely based on [`resolve_group_by`].

        [`resolve_group_by`]: https://github.com/pola-rs/polars/blob/cdd247aaba8db3332be0bd031e0f31bc3fc33f77/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs#L1125-L1227
        """
        obj = cls.__new__(cls)
        keys, schema_in = prepare_projection(grouper._keys, schema=context)
        obj._keys, obj._schema_in = keys, schema_in
        obj._key_names = tuple(e.name for e in keys)
        obj._aggs, _ = prepare_projection(grouper._aggs, obj.key_names, schema=schema_in)
        obj._schema = schema_in.select(keys).merge(schema_in.select(obj._aggs))
        obj._drop_null_keys = grouper._drop_null_keys
        return obj

    def requires_projection(self, *, allow_aliasing: bool = False) -> bool:
        """Return True is group keys contain anything that is not a column selection.

        Notes:
            If False is returned, we can just use the resolved key names as a fast-path to group.

        Arguments:
            allow_aliasing: If False (default), any aliasing is not considered to be column selection.
        """
        if not all(key.is_column(allow_aliasing=allow_aliasing) for key in self.keys):
            if self._drop_null_keys:
                msg = "drop_null_keys cannot be True when keys contains Expr or Series"
                raise NotImplementedError(msg)
            return True
        return False


class Resolved(GroupByResolver):
    """Compliant-level `GroupBy` resolver."""

    _drop_null_keys: bool = False


class Grouped(Grouper[Resolved]):
    """Compliant-level `GroupBy` helper."""

    _keys: Seq[ExprIR]
    _aggs: Seq[ExprIR]
    _drop_null_keys: bool = False

    @property
    def _resolver(self) -> type[Resolved]:
        return Resolved
