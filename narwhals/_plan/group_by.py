"""Refresher on `rust` impl.

- [`resolve_group_by`] has the dsl algo
  - Depends on some `expr_expansion` functions I've implemented
  - `group_by_dynamic` is there also (but not doing that)
  - ooooh [auto-implode]
- [`dsl_to_ir::to_alp_impl`] was the caller of ^^^^^



[`resolve_group_by`]: https://github.com/pola-rs/polars/blob/cdd247aaba8db3332be0bd031e0f31bc3fc33f77/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs#L1125-L1227
[auto-implode]: https://github.com/pola-rs/polars/blob/cdd247aaba8db3332be0bd031e0f31bc3fc33f77/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs#L1197-L1203
[`dsl_to_ir::to_alp_impl`]: https://github.com/pola-rs/polars/blob/cdd247aaba8db3332be0bd031e0f31bc3fc33f77/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs#L459-L509
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from narwhals._plan import _parse
from narwhals._plan.typing import DataFrameT

if TYPE_CHECKING:
    from narwhals._plan.expressions import ExprIR
    from narwhals._plan.typing import IntoExpr, OneOrIterable, Seq


class GroupBy(Generic[DataFrameT]):
    _frame: DataFrameT
    _keys: Seq[ExprIR]

    def __init__(self, frame: DataFrameT, keys: Seq[ExprIR], /) -> None:
        self._frame = frame
        self._keys = keys

    def agg(self, *aggs: OneOrIterable[IntoExpr], **named_aggs: IntoExpr) -> DataFrameT:
        exprs = _parse.parse_into_seq_of_expr_ir(*aggs, **named_aggs)  # noqa: F841
        raise NotImplementedError
