from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, overload

from narwhals._plan.compliant.typing import (
    ConcatT1,
    ConcatT2,
    EagerDataFrameT,
    EagerExprT_co,
    EagerScalarT_co,
    ExprT_co,
    FrameT,
    HasVersion,
    LazyExprT_co,
    LazyScalarT_co,
    ScalarT_co,
    SeriesT,
)
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Iterable

    from typing_extensions import TypeIs

    from narwhals._plan import expressions as ir
    from narwhals._plan.expressions import FunctionExpr, boolean, functions as F
    from narwhals._plan.expressions.ranges import DateRange, IntRange, LinearSpace
    from narwhals._plan.expressions.strings import ConcatStr
    from narwhals._plan.series import Series
    from narwhals.dtypes import IntegerType
    from narwhals.typing import ClosedInterval, ConcatMethod, NonNestedLiteral

Int64 = Version.MAIN.dtypes.Int64()


class CompliantNamespace(HasVersion, Protocol[FrameT, ExprT_co, ScalarT_co]):
    implementation: ClassVar[Implementation]

    @property
    def _expr(self) -> type[ExprT_co]: ...
    @property
    def _frame(self) -> type[FrameT]: ...
    @property
    def _scalar(self) -> type[ScalarT_co]: ...
    def all_horizontal(
        self, node: FunctionExpr[boolean.AllHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def any_horizontal(
        self, node: FunctionExpr[boolean.AnyHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def col(self, node: ir.Column, frame: FrameT, name: str) -> ExprT_co: ...
    def concat_str(
        self, node: FunctionExpr[ConcatStr], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def coalesce(
        self, node: FunctionExpr[F.Coalesce], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def date_range(
        self, node: ir.RangeExpr[DateRange], frame: FrameT, name: str
    ) -> ExprT_co: ...
    def int_range(
        self, node: ir.RangeExpr[IntRange], frame: FrameT, name: str
    ) -> ExprT_co: ...
    def linear_space(
        self, node: ir.RangeExpr[LinearSpace], frame: FrameT, name: str
    ) -> ExprT_co: ...
    def len(self, node: ir.Len, frame: FrameT, name: str) -> ScalarT_co: ...
    def lit(
        self, node: ir.Literal[Any], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def max_horizontal(
        self, node: FunctionExpr[F.MaxHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def mean_horizontal(
        self, node: FunctionExpr[F.MeanHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def min_horizontal(
        self, node: FunctionExpr[F.MinHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def sum_horizontal(
        self, node: FunctionExpr[F.SumHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...


# NOTE: `mypy` is wrong
# error: Invariant type variable "ConcatT2" used in protocol where covariant one is expected  [misc]
class Concat(Protocol[ConcatT1, ConcatT2]):  # type: ignore[misc]
    @overload
    def concat(self, items: Iterable[ConcatT1], *, how: ConcatMethod) -> ConcatT1: ...
    # Series only supports vertical publicly (like in polars)
    @overload
    def concat(
        self, items: Iterable[ConcatT2], *, how: Literal["vertical"]
    ) -> ConcatT2: ...
    def concat(
        self, items: Iterable[ConcatT1 | ConcatT2], *, how: ConcatMethod
    ) -> ConcatT1 | ConcatT2: ...


class EagerConcat(Concat[ConcatT1, ConcatT2], Protocol[ConcatT1, ConcatT2]):  # type: ignore[misc]
    def _concat_diagonal(self, items: Iterable[ConcatT1], /) -> ConcatT1: ...
    # Series can be used here to go from [Series, Series] -> DataFrame
    # but that is only available privately
    def _concat_horizontal(self, items: Iterable[ConcatT1 | ConcatT2], /) -> ConcatT1: ...
    def _concat_vertical(
        self, items: Iterable[ConcatT1 | ConcatT2], /
    ) -> ConcatT1 | ConcatT2: ...


class EagerNamespace(
    EagerConcat[EagerDataFrameT, SeriesT],
    CompliantNamespace[EagerDataFrameT, EagerExprT_co, EagerScalarT_co],
    Protocol[EagerDataFrameT, SeriesT, EagerExprT_co, EagerScalarT_co],
):
    @property
    def _dataframe(self) -> type[EagerDataFrameT]: ...
    @property
    def _frame(self) -> type[EagerDataFrameT]:
        return self._dataframe

    @property
    def _series(self) -> type[SeriesT]: ...
    def _is_dataframe(self, obj: Any) -> TypeIs[EagerDataFrameT]:
        return isinstance(obj, self._dataframe)

    def _is_series(self, obj: Any) -> TypeIs[SeriesT]:
        return isinstance(obj, self._series)

    def len(self, node: ir.Len, frame: EagerDataFrameT, name: str) -> EagerScalarT_co:
        return self._scalar.from_python(
            len(frame), name or node.name, dtype=None, version=frame.version
        )

    @overload
    def lit(
        self, node: ir.Literal[NonNestedLiteral], frame: EagerDataFrameT, name: str
    ) -> EagerScalarT_co: ...
    @overload
    def lit(
        self, node: ir.Literal[Series[Any]], frame: EagerDataFrameT, name: str
    ) -> EagerExprT_co: ...
    def lit(
        self, node: ir.Literal[Any], frame: EagerDataFrameT, name: str
    ) -> EagerExprT_co | EagerScalarT_co: ...
    def date_range_eager(
        self,
        start: dt.date,
        end: dt.date,
        interval: int = 1,
        *,
        closed: ClosedInterval = "both",
        name: str = "literal",
    ) -> SeriesT: ...
    def int_range_eager(
        self,
        start: int,
        end: int,
        step: int = 1,
        *,
        dtype: IntegerType = Int64,
        name: str = "literal",
    ) -> SeriesT: ...
    def linear_space_eager(
        self,
        start: float,
        end: float,
        num_samples: int,
        *,
        closed: ClosedInterval = "both",
        name: str = "literal",
    ) -> SeriesT: ...


class LazyNamespace(
    Concat[FrameT, FrameT],
    CompliantNamespace[FrameT, LazyExprT_co, LazyScalarT_co],
    Protocol[FrameT, LazyExprT_co, LazyScalarT_co],
):
    @property
    def _lazyframe(self) -> type[FrameT]: ...
    @property
    def _frame(self) -> type[FrameT]:
        return self._lazyframe
