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

from typing import TYPE_CHECKING, Any, Generic, NamedTuple

from narwhals._plan import _parse
from narwhals._plan._expansion import prepare_projection
from narwhals._plan.typing import DataFrameT

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._plan.expressions import ExprIR, NamedIR
    from narwhals._plan.schema import FrozenSchema, IntoFrozenSchema
    from narwhals._plan.typing import IntoExpr, OneOrIterable, Seq


class GroupBy(Generic[DataFrameT]):
    _frame: DataFrameT
    _keys: Seq[ExprIR]
    _drop_null_keys: bool

    def __init__(
        self, frame: DataFrameT, keys: Seq[ExprIR], /, *, drop_null_keys: bool = False
    ) -> None:
        self._frame = frame
        self._keys = keys
        self._drop_null_keys = drop_null_keys

    def agg(self, *aggs: OneOrIterable[IntoExpr], **named_aggs: IntoExpr) -> DataFrameT:
        frame = self._frame
        resolved = resolve_group_by(
            self._keys, _parse.parse_into_seq_of_expr_ir(*aggs, **named_aggs), frame
        )
        compliant = frame._compliant
        compliant_gb = compliant._group_by
        # Do we need to project first?
        if not all(key.is_column() for key in resolved.keys):
            if self._drop_null_keys:
                msg = "drop_null_keys cannot be True when keys contains Expr or Series"
                raise NotImplementedError(msg)
            grouped = compliant_gb.by_named_irs(compliant, resolved.keys)
        else:
            # If not, we can just use the resolved key names as a fast-path
            grouped = compliant_gb.by_names(
                compliant, resolved.keys_names, drop_null_keys=self._drop_null_keys
            )
        return self._frame._from_compliant(grouped.agg(resolved.aggs))

    def __iter__(self) -> Iterator[tuple[Any, DataFrameT]]:
        msg = "Not Implemented `GroupBy.__iter__`"
        raise NotImplementedError(msg)


class _TempGroupByStuff(NamedTuple):
    """Trying to organize info that's useful to keep from `resolve_group_by`.

    Important:
        Not a long-term thing!
    """

    keys: Seq[NamedIR]
    aggs: Seq[NamedIR]
    keys_names: Seq[str]
    result_schema: FrozenSchema


def resolve_group_by(
    input_keys: Seq[ExprIR], input_aggs: Seq[ExprIR], input_schema: IntoFrozenSchema
) -> _TempGroupByStuff:
    keys, schema = prepare_projection(input_keys, schema=input_schema)
    keys_names = tuple(e.name for e in keys)
    aggs, _ = prepare_projection(input_aggs, keys_names, schema=schema)
    result_schema = schema.select(keys).merge(schema.select(aggs))
    return _TempGroupByStuff(keys, aggs, keys_names, result_schema)
