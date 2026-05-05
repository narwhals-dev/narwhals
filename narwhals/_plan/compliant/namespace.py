from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from narwhals._plan.compliant import io, ranges
from narwhals._plan.compliant.typing import (
    EagerDataFrameT,
    EagerExprT_co,
    EagerScalarT_co,
    ExprT_co,
    FrameT,
    ScalarT_co,
    SeriesT_co,
)
from narwhals._plan.typing import NativeDataFrameT_co, NativeSeriesT_co
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._plan import expressions as ir
    from narwhals._plan.expressions import HorizontalExpr as HExpr, functions as F
    from narwhals._plan.expressions.strings import ConcatStr
    from narwhals._utils import Implementation, Version
    from narwhals.typing import PythonLiteral

Incomplete: TypeAlias = Any


# TODO @dangotbanned: Review what will replace following `*Classes`
class CompliantNamespace(
    io.ReadCsvSchema,
    io.ReadParquetSchema,
    ranges.LazyRangeGenerator[FrameT, ExprT_co],
    Protocol[FrameT, ExprT_co, ScalarT_co],
):
    """`[FrameT, ExprT_co, ScalarT_co]`.

    ## Notes of `FrameT` variance
    - An issue for `LazyRangeGenerator` if that can be either eager or lazy
    - Having `CompliantFrameV*` and using `Self` is fragile
    """

    implementation: ClassVar[Implementation]
    version: ClassVar[Version]

    @property
    def _expr(self) -> type[ExprT_co]: ...
    @property
    def _frame(self) -> type[FrameT]:
        """The invariance of `FrameT` is a big ol problemo."""
        ...

    @property
    def _scalar(self) -> type[ScalarT_co]: ...
    def all_horizontal(
        self, node: HExpr[ir.boolean.AllHorizontal], frame: FrameT, name: str, /
    ) -> ExprT_co | ScalarT_co: ...
    def any_horizontal(
        self, node: HExpr[ir.boolean.AnyHorizontal], frame: FrameT, name: str, /
    ) -> ExprT_co | ScalarT_co: ...
    def col(self, node: ir.Column, frame: FrameT, name: str, /) -> ExprT_co: ...
    def concat_str(
        self, node: HExpr[ConcatStr], frame: FrameT, name: str, /
    ) -> ExprT_co | ScalarT_co: ...
    def coalesce(
        self, node: HExpr[F.Coalesce], frame: FrameT, name: str, /
    ) -> ExprT_co | ScalarT_co: ...
    def len(self, node: ir.Len, frame: FrameT, name: str, /) -> ScalarT_co: ...
    def lit(
        self, node: ir.Lit[PythonLiteral], frame: FrameT, name: str, /
    ) -> ScalarT_co: ...
    def max_horizontal(
        self, node: HExpr[F.MaxHorizontal], frame: FrameT, name: str, /
    ) -> ExprT_co | ScalarT_co: ...
    def mean_horizontal(
        self, node: HExpr[F.MeanHorizontal], frame: FrameT, name: str, /
    ) -> ExprT_co | ScalarT_co: ...
    def min_horizontal(
        self, node: HExpr[F.MinHorizontal], frame: FrameT, name: str, /
    ) -> ExprT_co | ScalarT_co: ...
    def sum_horizontal(
        self, node: HExpr[F.SumHorizontal], frame: FrameT, name: str, /
    ) -> ExprT_co | ScalarT_co: ...

    def __narwhals_namespace__(self) -> Self:
        return self

    def from_named_ir(
        self, named_ir: ir.NamedIR, frame: FrameT, /
    ) -> ExprT_co | ScalarT_co:
        return named_ir.dispatch(self, frame)

    def from_ir(
        self, node: ir.ExprIR, frame: FrameT, name: str, /
    ) -> ExprT_co | ScalarT_co:
        return node.dispatch(self, frame, name)

    def __narwhals_expr_prepare__(self) -> ExprT_co:
        return self._expr.__new__(self._expr)

    # NOTE: will reduce direct calls to `*Namespace._<compliant-type>`
    from_native: not_implemented = not_implemented()


class EagerNamespace(
    ranges.EagerRangeGenerator[NativeSeriesT_co],
    CompliantNamespace[EagerDataFrameT, EagerExprT_co, EagerScalarT_co],
    Protocol[
        EagerDataFrameT,
        SeriesT_co,
        EagerExprT_co,
        EagerScalarT_co,
        NativeDataFrameT_co,
        NativeSeriesT_co,
    ],
):
    """`[EagerDataFrameT, SeriesT_co, EagerExprT_co, EagerScalarT_co, NativeDataFrameT_co, NativeSeriesT_co]`.

    ## Important
    Trying to
    - reduce the number of type params
    - ensure most are covariant
    - rely on native types when possible
    """

    @property
    def _dataframe(self) -> type[EagerDataFrameT]: ...
    @property
    def _frame(self) -> type[EagerDataFrameT]:  # pragma: no cover
        return self._dataframe

    @property
    def _series(self) -> type[SeriesT_co]: ...
    def lit_series(
        self, node: ir.LitSeries[Any], frame: EagerDataFrameT, name: str, /
    ) -> EagerExprT_co: ...
