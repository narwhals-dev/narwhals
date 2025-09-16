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

from typing import TYPE_CHECKING, Generic

from narwhals._plan import _parse
from narwhals._plan._expansion import (
    ensure_valid_exprs,
    into_named_irs,
    rewrite_projections,
)
from narwhals._plan.schema import FrozenSchema, freeze_schema
from narwhals._plan.typing import DataFrameT

if TYPE_CHECKING:
    from narwhals._plan.expressions import ExprIR, NamedIR
    from narwhals._plan.typing import IntoExpr, OneOrIterable, Seq
    from narwhals.schema import Schema


class GroupBy(Generic[DataFrameT]):
    _frame: DataFrameT
    _keys: Seq[ExprIR]

    def __init__(self, frame: DataFrameT, keys: Seq[ExprIR], /) -> None:
        self._frame = frame
        self._keys = keys

    def agg(self, *aggs: OneOrIterable[IntoExpr], **named_aggs: IntoExpr) -> DataFrameT:
        keys_named_irs, aggs_named_irs, result_schema = resolve_group_by(  # noqa: RUF059
            self._keys,
            _parse.parse_into_seq_of_expr_ir(*aggs, **named_aggs),
            self._frame.schema,
        )
        msg = "`GroupBy.agg` needs a compliant-level to dispatch to"
        raise NotImplementedError(msg)


def resolve_group_by(
    input_keys: Seq[ExprIR], input_aggs: Seq[ExprIR], schema: Schema
) -> tuple[Seq[NamedIR], Seq[NamedIR], FrozenSchema]:
    input_schema = freeze_schema(schema)

    # "Initialize schema from keys"
    keys = rewrite_projections(input_keys, keys=(), schema=input_schema)
    key_names = ensure_valid_exprs(keys, input_schema)
    keys_named_irs = into_named_irs(keys, key_names)
    output_schema = input_schema._select(keys_named_irs)

    # "Add aggregation column(s)"  # noqa: ERA001
    aggs = rewrite_projections(input_aggs, keys=key_names, schema=input_schema)
    aggs_names = ensure_valid_exprs(aggs, input_schema)
    aggs_named_irs = into_named_irs(aggs, aggs_names)
    aggs_schema = input_schema._select(aggs_named_irs)

    # "Coerce aggregation column(s) into List unless not needed (auto-implode)"  # noqa: ERA001
    # TODO @dangotbanned: seems to just be a schema transform, maybe not important for now?

    # "Final output_schema"
    result_schema = output_schema.merge(aggs_schema)

    # "Make sure aggregation columns do not contain keys or index columns"
    # TODO @dangotbanned: Probably just the keys part?
    # *index columns* seems to be rolling/dynamic only
    return keys_named_irs, aggs_named_irs, result_schema
