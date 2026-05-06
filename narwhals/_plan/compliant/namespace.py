from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol

from narwhals._plan.compliant import io, ranges, typing as ct
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan import expressions as ir
    from narwhals._plan.expressions import HorizontalExpr as HExpr, functions as F
    from narwhals._plan.expressions.strings import ConcatStr
    from narwhals._utils import Implementation, Version
    from narwhals.typing import PythonLiteral


# TODO @dangotbanned: Review what will replace following `*Classes`
class CompliantNamespace(
    io.ReadSchema,
    ranges.LazyRangeGenerator[ct.FrameT, ct.ExprT_co],
    Protocol[ct.FrameT, ct.ExprT_co, ct.ScalarT_co],
):
    """`[FrameT, ExprT_co, ScalarT_co]`.

    ## Notes of `FrameT` variance
    - An issue for `LazyRangeGenerator` if that can be either eager or lazy
    - Having `CompliantFrameV*` and using `Self` is fragile
    """

    __slots__ = ()

    implementation: ClassVar[Implementation]
    version: ClassVar[Version]

    @property
    def _expr(self) -> type[ct.ExprT_co]: ...
    @property
    def _frame(self) -> type[ct.FrameT]:
        """The invariance of `FrameT` is a big ol problemo."""
        ...

    @property
    def _scalar(self) -> type[ct.ScalarT_co]: ...
    def all_horizontal(
        self, node: HExpr[ir.boolean.AllHorizontal], frame: ct.FrameT, name: str, /
    ) -> ct.ExprT_co | ct.ScalarT_co: ...
    def any_horizontal(
        self, node: HExpr[ir.boolean.AnyHorizontal], frame: ct.FrameT, name: str, /
    ) -> ct.ExprT_co | ct.ScalarT_co: ...
    def col(self, node: ir.Column, frame: ct.FrameT, name: str, /) -> ct.ExprT_co: ...
    def concat_str(
        self, node: HExpr[ConcatStr], frame: ct.FrameT, name: str, /
    ) -> ct.ExprT_co | ct.ScalarT_co: ...
    def coalesce(
        self, node: HExpr[F.Coalesce], frame: ct.FrameT, name: str, /
    ) -> ct.ExprT_co | ct.ScalarT_co: ...
    def len_star(self, node: ir.Len, frame: ct.FrameT, name: str, /) -> ct.ScalarT_co: ...
    def lit(
        self, node: ir.Lit[PythonLiteral], frame: ct.FrameT, name: str, /
    ) -> ct.ScalarT_co: ...
    def max_horizontal(
        self, node: HExpr[F.MaxHorizontal], frame: ct.FrameT, name: str, /
    ) -> ct.ExprT_co | ct.ScalarT_co: ...
    def mean_horizontal(
        self, node: HExpr[F.MeanHorizontal], frame: ct.FrameT, name: str, /
    ) -> ct.ExprT_co | ct.ScalarT_co: ...
    def min_horizontal(
        self, node: HExpr[F.MinHorizontal], frame: ct.FrameT, name: str, /
    ) -> ct.ExprT_co | ct.ScalarT_co: ...
    def sum_horizontal(
        self, node: HExpr[F.SumHorizontal], frame: ct.FrameT, name: str, /
    ) -> ct.ExprT_co | ct.ScalarT_co: ...

    def __narwhals_namespace__(self) -> Self:
        return self

    def from_named_ir(
        self, named_ir: ir.NamedIR, frame: ct.FrameT, /
    ) -> ct.ExprT_co | ct.ScalarT_co:
        return named_ir.dispatch(self, frame)

    def from_ir(
        self, node: ir.ExprIR, frame: ct.FrameT, name: str, /
    ) -> ct.ExprT_co | ct.ScalarT_co:
        return node.dispatch(self, frame, name)

    def __narwhals_expr_prepare__(self) -> ct.ExprT_co:
        return self._expr.__new__(self._expr)

    # NOTE: will reduce direct calls to `*Namespace._<compliant-type>`
    from_native: not_implemented = not_implemented()


class EagerNamespace(
    CompliantNamespace[ct.EagerDataFrameT, ct.EagerExprT_co, ct.EagerScalarT_co],
    Protocol[ct.EagerDataFrameT, ct.SeriesT_co, ct.EagerExprT_co, ct.EagerScalarT_co],
):
    """`[EagerDataFrameT, SeriesT_co, EagerExprT_co, EagerScalarT_co]`.

    ## Important
    Trying to
    - reduce the number of type params
    - ensure most are covariant
    - rely on native types when possible
    """

    __slots__ = ()

    @property
    def _dataframe(self) -> type[ct.EagerDataFrameT]: ...
    @property
    def _frame(self) -> type[ct.EagerDataFrameT]:  # pragma: no cover
        return self._dataframe

    @property
    def _series(self) -> type[ct.SeriesT_co]: ...


def namespace(obj: ct.SupportsNarwhalsNamespace[ct.NamespaceT_co], /) -> ct.NamespaceT_co:
    """Get the compliant namespace from `obj`."""
    return obj.__narwhals_namespace__()
