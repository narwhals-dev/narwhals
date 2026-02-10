from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from narwhals._plan._parse import parse_into_seq_of_expr_ir
from narwhals._plan.compliant.group_by import GroupByResolver as Resolved, Grouper
from narwhals._plan.typing import DataFrameT, LazyFrameT

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

    from narwhals._plan.expressions import ExprIR
    from narwhals._plan.typing import IntoExpr, OneOrIterable, Seq


class GroupBy(Generic[DataFrameT]):
    _frame: DataFrameT
    _grouper: Grouped

    def __init__(self, frame: DataFrameT, grouper: Grouped, /) -> None:
        self._frame = frame
        self._grouper = grouper

    def agg(self, *aggs: OneOrIterable[IntoExpr], **named_aggs: IntoExpr) -> DataFrameT:
        frame = self._frame
        return frame._with_compliant(
            self._grouper.agg(*aggs, **named_aggs)
            .resolve(frame)
            .evaluate(frame._compliant)
        )

    def __iter__(self) -> Iterator[tuple[Any, DataFrameT]]:
        frame = self._frame
        resolver = self._grouper.agg().resolve(frame)
        for key, df in frame._compliant.group_by_resolver(resolver):
            yield key, frame._with_compliant(df)


class LazyGroupBy(Generic[LazyFrameT]):  # pragma: no cover
    _frame: LazyFrameT
    _keys: Seq[ExprIR]

    def __init__(
        self,
        frame: LazyFrameT,
        *by: OneOrIterable[IntoExpr],
        drop_null_keys: bool = False,
        **named_by: IntoExpr,
    ) -> None:
        if drop_null_keys:
            msg = f"TODO: LazyFrame.group_by({drop_null_keys=})"
            raise NotImplementedError(msg)
        self._frame = frame
        self._keys = parse_into_seq_of_expr_ir(*by, **named_by)

    def agg(self, *aggs: OneOrIterable[IntoExpr], **named_aggs: IntoExpr) -> LazyFrameT:
        frame = self._frame
        aggs_ir = parse_into_seq_of_expr_ir(*aggs, **named_aggs)
        return frame._with_lp(frame._plan.group_by(self._keys, aggs_ir))


class Grouped(Grouper["Resolved"]):
    """Narwhals-level `GroupBy` builder."""

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
        obj._keys = parse_into_seq_of_expr_ir(*by, **named_by)
        obj._drop_null_keys = drop_null_keys
        return obj

    def agg(self, *aggs: OneOrIterable[IntoExpr], **named_aggs: IntoExpr) -> Self:
        self._aggs = parse_into_seq_of_expr_ir(*aggs, **named_aggs)
        return self

    @property
    def _resolver(self) -> type[Resolved]:
        return Resolved

    def to_group_by(self, frame: DataFrameT, /) -> GroupBy[DataFrameT]:
        return GroupBy(frame, self)
