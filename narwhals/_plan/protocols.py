"""TODO: Split this module up into `narwhals._plan.compliant.*`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

from narwhals._plan.compliant.column import EagerBroadcast, SupportsBroadcast
from narwhals._plan.compliant.typing import (
    ColumnT_co,
    FrameT_contra,
    LengthT,
    NamespaceT_co,
    R_co,
    SeriesT,
    SeriesT_co,
    StoresVersion,
)
from narwhals._plan.typing import (
    IntoExpr,
    NativeDataFrameT,
    NativeFrameT,
    NativeSeriesT,
    Seq,
)
from narwhals._utils import Version

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from typing_extensions import Self

    from narwhals._plan import expressions as ir
    from narwhals._plan.compliant.group_by import (
        CompliantGroupBy,
        DataFrameGroupBy,
        EagerDataFrameGroupBy,
        GroupByResolver,
        Grouped,
    )
    from narwhals._plan.compliant.namespace import EagerNamespace
    from narwhals._plan.dataframe import BaseFrame, DataFrame
    from narwhals._plan.expressions import (
        BinaryExpr,
        FunctionExpr,
        NamedIR,
        aggregation as agg,
        boolean,
        functions as F,
    )
    from narwhals._plan.expressions.boolean import IsBetween, IsFinite, IsNan, IsNull, Not
    from narwhals._plan.options import SortMultipleOptions
    from narwhals._plan.typing import OneOrIterable
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType, IntoSchema, PythonLiteral


class ExprDispatch(StoresVersion, Protocol[FrameT_contra, R_co, NamespaceT_co]):
    @classmethod
    def from_ir(cls, node: ir.ExprIR, frame: FrameT_contra, name: str) -> R_co:
        obj = cls.__new__(cls)
        obj._version = frame.version
        return node.dispatch(obj, frame, name)

    @classmethod
    def from_named_ir(cls, named_ir: NamedIR[ir.ExprIR], frame: FrameT_contra) -> R_co:
        return cls.from_ir(named_ir.expr, frame, named_ir.name)

    # NOTE: Needs to stay `covariant` and never be used as a parameter
    def __narwhals_namespace__(self) -> NamespaceT_co: ...


class CompliantExpr(StoresVersion, Protocol[FrameT_contra, SeriesT_co]):
    """Everything common to `Expr`/`Series` and `Scalar` literal values."""

    _evaluated: Any
    """Compliant or native value."""

    @property
    def name(self) -> str: ...
    @classmethod
    def from_native(
        cls, native: Any, name: str = "", /, version: Version = Version.MAIN
    ) -> Self: ...
    def _with_native(self, native: Any, name: str, /) -> Self:
        return self.from_native(native, name or self.name, self.version)

    # series & scalar
    def abs(self, node: FunctionExpr[F.Abs], frame: FrameT_contra, name: str) -> Self: ...
    def cast(self, node: ir.Cast, frame: FrameT_contra, name: str) -> Self: ...
    def pow(self, node: FunctionExpr[F.Pow], frame: FrameT_contra, name: str) -> Self: ...
    def not_(self, node: FunctionExpr[Not], frame: FrameT_contra, name: str) -> Self: ...
    def fill_null(
        self, node: FunctionExpr[F.FillNull], frame: FrameT_contra, name: str
    ) -> Self: ...
    def is_between(
        self, node: FunctionExpr[IsBetween], frame: FrameT_contra, name: str
    ) -> Self: ...
    def is_finite(
        self, node: FunctionExpr[IsFinite], frame: FrameT_contra, name: str
    ) -> Self: ...
    def is_nan(
        self, node: FunctionExpr[IsNan], frame: FrameT_contra, name: str
    ) -> Self: ...
    def is_null(
        self, node: FunctionExpr[IsNull], frame: FrameT_contra, name: str
    ) -> Self: ...
    def binary_expr(self, node: BinaryExpr, frame: FrameT_contra, name: str) -> Self: ...
    def ternary_expr(
        self, node: ir.TernaryExpr, frame: FrameT_contra, name: str
    ) -> Self: ...
    def over(self, node: ir.WindowExpr, frame: FrameT_contra, name: str) -> Self: ...
    # NOTE: `Scalar` is returned **only** for un-partitioned `OrderableAggExpr`
    # e.g. `nw.col("a").first().over(order_by="b")`
    def over_ordered(
        self, node: ir.OrderedWindowExpr, frame: FrameT_contra, name: str
    ) -> Self | CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def map_batches(
        self, node: ir.AnonymousExpr, frame: FrameT_contra, name: str
    ) -> Self: ...
    def rolling_expr(
        self, node: ir.RollingExpr, frame: FrameT_contra, name: str
    ) -> Self: ...
    # series only (section 3)
    def sort(self, node: ir.Sort, frame: FrameT_contra, name: str) -> Self: ...
    def sort_by(self, node: ir.SortBy, frame: FrameT_contra, name: str) -> Self: ...
    def filter(self, node: ir.Filter, frame: FrameT_contra, name: str) -> Self: ...
    # series -> scalar
    def first(
        self, node: agg.First, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def last(
        self, node: agg.Last, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def arg_min(
        self, node: agg.ArgMin, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def arg_max(
        self, node: agg.ArgMax, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def sum(
        self, node: agg.Sum, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def n_unique(
        self, node: agg.NUnique, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def std(
        self, node: agg.Std, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def var(
        self, node: agg.Var, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def quantile(
        self, node: agg.Quantile, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def count(
        self, node: agg.Count, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def len(
        self, node: agg.Len, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def max(
        self, node: agg.Max, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def mean(
        self, node: agg.Mean, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def median(
        self, node: agg.Median, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def min(
        self, node: agg.Min, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def all(
        self, node: FunctionExpr[boolean.All], frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def any(
        self, node: FunctionExpr[boolean.Any], frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...


class CompliantScalar(
    CompliantExpr[FrameT_contra, SeriesT_co], Protocol[FrameT_contra, SeriesT_co]
):
    _name: str

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_python(
        cls,
        value: PythonLiteral,
        name: str = "literal",
        /,
        *,
        dtype: IntoDType | None,
        version: Version,
    ) -> Self: ...
    def _with_evaluated(self, evaluated: Any, name: str) -> Self:
        """Expr is based on a series having these via accessors, but a scalar needs to keep passing through."""
        cls = type(self)
        obj = cls.__new__(cls)
        obj._evaluated = evaluated
        obj._name = name or self.name
        obj._version = self.version
        return obj

    def max(self, node: agg.Max, frame: FrameT_contra, name: str) -> Self:
        """Returns self."""
        return self._with_evaluated(self._evaluated, name)

    def min(self, node: agg.Min, frame: FrameT_contra, name: str) -> Self:
        """Returns self."""
        return self._with_evaluated(self._evaluated, name)

    def sum(self, node: agg.Sum, frame: FrameT_contra, name: str) -> Self:
        """Returns self."""
        return self._with_evaluated(self._evaluated, name)

    def first(self, node: agg.First, frame: FrameT_contra, name: str) -> Self:
        """Returns self."""
        return self._with_evaluated(self._evaluated, name)

    def last(self, node: agg.Last, frame: FrameT_contra, name: str) -> Self:
        """Returns self."""
        return self._with_evaluated(self._evaluated, name)

    def _cast_float(self, node: ir.ExprIR, frame: FrameT_contra, name: str) -> Self:
        """`polars` interpolates a single scalar as a float."""
        dtype = self.version.dtypes.Float64()
        return self.cast(node.cast(dtype), frame, name)

    def mean(self, node: agg.Mean, frame: FrameT_contra, name: str) -> Self:
        return self._cast_float(node.expr, frame, name)

    def median(self, node: agg.Median, frame: FrameT_contra, name: str) -> Self:
        return self._cast_float(node.expr, frame, name)

    def quantile(self, node: agg.Quantile, frame: FrameT_contra, name: str) -> Self:
        return self._cast_float(node.expr, frame, name)

    def n_unique(self, node: agg.NUnique, frame: FrameT_contra, name: str) -> Self:
        """Returns 1."""
        ...

    def std(self, node: agg.Std, frame: FrameT_contra, name: str) -> Self:
        """Returns null."""
        ...

    def var(self, node: agg.Var, frame: FrameT_contra, name: str) -> Self:
        """Returns null."""
        ...

    def arg_min(self, node: agg.ArgMin, frame: FrameT_contra, name: str) -> Self:
        """Returns 0."""
        ...

    def arg_max(self, node: agg.ArgMax, frame: FrameT_contra, name: str) -> Self:
        """Returns 0."""
        ...

    def count(self, node: agg.Count, frame: FrameT_contra, name: str) -> Self:
        """Returns 0 if null, else 1."""
        ...

    def len(self, node: agg.Len, frame: FrameT_contra, name: str) -> Self:
        """Returns 1."""
        ...

    def sort(self, node: ir.Sort, frame: FrameT_contra, name: str) -> Self:
        return self._with_evaluated(self._evaluated, name)

    def sort_by(self, node: ir.SortBy, frame: FrameT_contra, name: str) -> Self:
        return self._with_evaluated(self._evaluated, name)

    # NOTE: `Filter` behaves the same, (maybe) no need to override


class EagerExpr(
    EagerBroadcast[SeriesT],
    CompliantExpr[FrameT_contra, SeriesT],
    Protocol[FrameT_contra, SeriesT],
): ...


class LazyExpr(
    SupportsBroadcast[SeriesT, LengthT],
    CompliantExpr[FrameT_contra, SeriesT],
    Protocol[FrameT_contra, SeriesT, LengthT],
): ...


class EagerScalar(
    CompliantScalar[FrameT_contra, SeriesT],
    EagerExpr[FrameT_contra, SeriesT],
    Protocol[FrameT_contra, SeriesT],
):
    def __len__(self) -> int:
        return 1

    def to_python(self) -> PythonLiteral: ...


class LazyScalar(
    CompliantScalar[FrameT_contra, SeriesT],
    LazyExpr[FrameT_contra, SeriesT, LengthT],
    Protocol[FrameT_contra, SeriesT, LengthT],
): ...


class CompliantBaseFrame(StoresVersion, Protocol[ColumnT_co, NativeFrameT]):
    _native: NativeFrameT

    def __narwhals_namespace__(self) -> Any: ...
    @property
    def _group_by(self) -> type[CompliantGroupBy[Self]]: ...
    @property
    def native(self) -> NativeFrameT:
        return self._native

    @property
    def columns(self) -> list[str]: ...
    def to_narwhals(self) -> BaseFrame[NativeFrameT]: ...
    @classmethod
    def from_native(cls, native: NativeFrameT, /, version: Version) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        obj._version = version
        return obj

    def _with_native(self, native: NativeFrameT) -> Self:
        return self.from_native(native, self.version)

    @property
    def schema(self) -> Mapping[str, DType]: ...
    def _evaluate_irs(
        self, nodes: Iterable[NamedIR[ir.ExprIR]], /
    ) -> Iterator[ColumnT_co]: ...
    def select(self, irs: Seq[NamedIR]) -> Self: ...
    def select_names(self, *column_names: str) -> Self: ...
    def with_columns(self, irs: Seq[NamedIR]) -> Self: ...
    def sort(self, by: Seq[NamedIR], options: SortMultipleOptions) -> Self: ...
    def drop(self, columns: Sequence[str], *, strict: bool = True) -> Self: ...
    def drop_nulls(self, subset: Sequence[str] | None) -> Self: ...


class CompliantDataFrame(
    CompliantBaseFrame[SeriesT, NativeDataFrameT],
    Protocol[SeriesT, NativeDataFrameT, NativeSeriesT],
):
    @property
    def _group_by(self) -> type[DataFrameGroupBy[Self]]: ...
    @property
    def _grouper(self) -> type[Grouped]:
        from narwhals._plan.compliant.group_by import Grouped

        return Grouped

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], /, *, schema: IntoSchema | None = None
    ) -> Self: ...
    def group_by_agg(
        self, by: OneOrIterable[IntoExpr], aggs: OneOrIterable[IntoExpr], /
    ) -> Self:
        """Compliant-level `group_by(by).agg(agg)`, allows `Expr`."""
        return self._grouper.by(by).agg(aggs).resolve(self).evaluate(self)

    def group_by_names(self, names: Seq[str], /) -> DataFrameGroupBy[Self]:
        """Compliant-level `group_by`, allowing only `str` keys."""
        return self._group_by.by_names(self, names)

    def group_by_resolver(self, resolver: GroupByResolver, /) -> DataFrameGroupBy[Self]:
        """Narwhals-level resolved `group_by`.

        `keys`, `aggs` are already parsed and projections planned.
        """
        return self._group_by.from_resolver(self, resolver)

    def to_narwhals(self) -> DataFrame[NativeDataFrameT, NativeSeriesT]: ...
    @overload
    def to_dict(self, *, as_series: Literal[True]) -> dict[str, SeriesT]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    @overload
    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, SeriesT] | dict[str, list[Any]]: ...
    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, SeriesT] | dict[str, list[Any]]: ...
    def __len__(self) -> int: ...
    def with_row_index(self, name: str) -> Self: ...
    def row(self, index: int) -> tuple[Any, ...]: ...


class EagerDataFrame(
    CompliantDataFrame[SeriesT, NativeDataFrameT, NativeSeriesT],
    Protocol[SeriesT, NativeDataFrameT, NativeSeriesT],
):
    @property
    def _group_by(self) -> type[EagerDataFrameGroupBy[Self]]: ...
    def __narwhals_namespace__(self) -> EagerNamespace[Self, SeriesT, Any, Any]: ...
    def select(self, irs: Seq[NamedIR]) -> Self:
        return self.__narwhals_namespace__()._concat_horizontal(self._evaluate_irs(irs))

    def with_columns(self, irs: Seq[NamedIR]) -> Self:
        return self.__narwhals_namespace__()._concat_horizontal(self._evaluate_irs(irs))
