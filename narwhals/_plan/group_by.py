"""Refresher on `rust` impl.

- [`resolve_group_by`] has the dsl algo
  - Depends on some `expr_expansion` functions I've implemented
  - `group_by_dynamic` is there also (but not doing that)
  - ooooh [auto-implode]
- [`dsl_to_ir::to_alp_impl`] was the caller of ^^^^^
- Misc recent important PRs
  - `1.32.1`
    - [Remove `Context` from logical layer]
  - `1.32.0`
    - [Make `Selector` a concrete part of the DSL]

[`resolve_group_by`]: https://github.com/pola-rs/polars/blob/cdd247aaba8db3332be0bd031e0f31bc3fc33f77/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs#L1125-L1227
[auto-implode]: https://github.com/pola-rs/polars/blob/cdd247aaba8db3332be0bd031e0f31bc3fc33f77/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs#L1197-L1203
[`dsl_to_ir::to_alp_impl`]: https://github.com/pola-rs/polars/blob/cdd247aaba8db3332be0bd031e0f31bc3fc33f77/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs#L459-L509
[Remove `Context` from logical layer]: https://github.com/pola-rs/polars/pull/23863
[Make `Selector` a concrete part of the DSL]: https://github.com/pola-rs/polars/pull/23351
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Protocol

from narwhals._plan import _parse
from narwhals._plan._expansion import prepare_projection
from narwhals._plan.typing import DataFrameT
from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

    from narwhals._plan.expressions import ExprIR, NamedIR
    from narwhals._plan.schema import FrozenSchema, IntoFrozenSchema
    from narwhals._plan.typing import IntoExpr, OneOrIterable, Seq

ResolverT_co = TypeVar("ResolverT_co", bound="Resolver", covariant=True)


class GroupBy(Generic[DataFrameT]):
    _frame: DataFrameT
    _grouper: Grouped

    def __init__(self, frame: DataFrameT, grouper: Grouped, /) -> None:
        self._frame = frame
        self._grouper = grouper

    def agg(self, *aggs: OneOrIterable[IntoExpr], **named_aggs: IntoExpr) -> DataFrameT:
        frame = self._frame
        resolved = self._grouper.agg(*aggs, **named_aggs).resolve(frame)
        compliant = frame._compliant
        compliant_gb = compliant._group_by
        if resolved.requires_projection():
            grouped = compliant_gb.by_named_irs(compliant, resolved.keys)
        else:
            grouped = compliant_gb.by_names(
                compliant, resolved.key_names, drop_null_keys=resolved._drop_null_keys
            )
        return self._frame._from_compliant(grouped.agg(resolved.aggs))

    def __iter__(self) -> Iterator[tuple[Any, DataFrameT]]:
        msg = "Not Implemented `GroupBy.__iter__`"
        raise NotImplementedError(msg)


class Grouper(Protocol[ResolverT_co]):
    """Revised interface focused on the state change + expression projections.

    - Uses `Expr` everywhere (no need to duplicate layers)
    - Resolver only needs schema (neither needs a frame, but can use one to get `schema`)
    """

    _keys: Seq[ExprIR]
    _aggs: Seq[ExprIR]
    _drop_null_keys: bool

    @classmethod
    def by(
        cls,
        *by: OneOrIterable[IntoExpr],
        drop_null_keys: bool = False,
        **named_by: IntoExpr,
    ) -> Self:
        obj = cls.__new__(cls)
        obj._keys = _parse.parse_into_seq_of_expr_ir(*by, **named_by)
        obj._drop_null_keys = drop_null_keys
        return obj

    def agg(self, *aggs: OneOrIterable[IntoExpr], **named_aggs: IntoExpr) -> Self:
        self._aggs = _parse.parse_into_seq_of_expr_ir(*aggs, **named_aggs)
        return self

    @property
    def _resolver(self) -> type[ResolverT_co]: ...

    def resolve(self, context: IntoFrozenSchema, /) -> ResolverT_co:
        return self._resolver.from_grouper(self, context)


class Resolver(Protocol):
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
        return self._key_names

    @classmethod
    def from_grouper(cls, grouper: Grouper[Self], context: IntoFrozenSchema, /) -> Self:
        obj = cls.__new__(cls)
        keys, schema_in = prepare_projection(grouper._keys, schema=context)
        obj._keys, obj._schema_in = keys, schema_in
        obj._key_names = tuple(e.name for e in keys)
        obj._aggs, _ = prepare_projection(grouper._aggs, obj._key_names, schema=schema_in)
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


class Grouped(Grouper["Resolved"]):
    """Narwhals-level `GroupBy` builder."""

    _keys: Seq[ExprIR]
    _aggs: Seq[ExprIR]
    _drop_null_keys: bool

    @property
    def _resolver(self) -> type[Resolved]:
        return Resolved

    def to_group_by(self, frame: DataFrameT, /) -> GroupBy[DataFrameT]:
        return GroupBy(frame, self)


class Resolved(Resolver):
    """Narwhals-level `GroupBy` resolver."""

    _schema_in: FrozenSchema
    _keys: Seq[NamedIR]
    _aggs: Seq[NamedIR]
    _key_names: Seq[str]
    _schema: FrozenSchema
    _drop_null_keys: bool
