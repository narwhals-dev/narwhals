from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from narwhals._plan.compliant import io, ranges
from narwhals._plan.compliant.concat import ConcatDataFrame, ConcatSeriesHorizontal
from narwhals._plan.compliant.typing import (
    EagerDataFrameT,
    EagerExprT_co,
    EagerScalarT_co,
    ExprT_co,
    FrameT,
    HasVersion,
    ScalarT_co,
    SeriesT_co,
)
from narwhals._plan.typing import NativeDataFrameT, NativeSeriesT_co

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan import expressions as ir
    from narwhals._plan.expressions import (
        HorizontalExpr as HExpr,
        boolean,
        functions as F,
    )
    from narwhals._plan.expressions.strings import ConcatStr
    from narwhals._utils import Implementation
    from narwhals.typing import PythonLiteral

Incomplete: TypeAlias = Any


# TODO @dangotbanned: Define `CompliantNamespace().from_native`
# - will reduce direct calls to `*Namespace._<compliant-type>`
class CompliantNamespace(
    HasVersion,
    # NOTE: Using `FrameT` in `LazyRangeGenerator` *could* be an issue if that can be either eager or lazy
    ranges.LazyRangeGenerator[FrameT, ExprT_co],
    Protocol[FrameT, ExprT_co, ScalarT_co],
):
    """`[FrameT, ExprT_co, ScalarT_co]`."""

    implementation: ClassVar[Implementation]

    @property
    def _expr(self) -> type[ExprT_co]: ...
    @property
    def _frame(self) -> type[FrameT]: ...
    @property
    def _scalar(self) -> type[ScalarT_co]: ...
    def all_horizontal(
        self, node: HExpr[boolean.AllHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def any_horizontal(
        self, node: HExpr[boolean.AnyHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def col(self, node: ir.Column, frame: FrameT, name: str) -> ExprT_co: ...
    def concat_str(
        self, node: HExpr[ConcatStr], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def coalesce(
        self, node: HExpr[F.Coalesce], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def len(self, node: ir.Len, frame: FrameT, name: str) -> ScalarT_co: ...
    def lit(
        self, node: ir.Lit[PythonLiteral], frame: FrameT, name: str
    ) -> ScalarT_co: ...
    def max_horizontal(
        self, node: HExpr[F.MaxHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def mean_horizontal(
        self, node: HExpr[F.MeanHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def min_horizontal(
        self, node: HExpr[F.MinHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def sum_horizontal(
        self, node: HExpr[F.SumHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...


class EagerNamespace(
    ranges.EagerRangeGenerator[NativeSeriesT_co],
    io.LazyInput[Incomplete],
    io.EagerInput[EagerDataFrameT],
    ConcatDataFrame[NativeDataFrameT, Incomplete],
    ConcatSeriesHorizontal[NativeDataFrameT, Incomplete],
    CompliantNamespace[EagerDataFrameT, EagerExprT_co, EagerScalarT_co],
    Protocol[
        EagerDataFrameT,
        SeriesT_co,
        EagerExprT_co,
        EagerScalarT_co,
        NativeDataFrameT,
        NativeSeriesT_co,
    ],
):
    """`[EagerDataFrameT, SeriesT_co, EagerExprT_co, EagerScalarT_co, NativeDataFrameT, NativeSeriesT_co]`.

    ## Important
    Trying to
    - reduce the number of type params
    - ensure most are covariant
    - rely on native types when possible
    """

    @property
    def _dataframe(self) -> type[EagerDataFrameT]: ...
    @property
    def _frame(self) -> type[EagerDataFrameT]:
        return self._dataframe

    @property
    def _series(self) -> type[SeriesT_co]: ...

    def len(self, node: ir.Len, frame: EagerDataFrameT, name: str) -> EagerScalarT_co:
        return self._scalar.from_python(
            len(frame), name or node.name, dtype=None, version=frame.version
        )

    def lit_series(
        self, node: ir.LitSeries[Any], frame: EagerDataFrameT, name: str
    ) -> EagerExprT_co: ...
