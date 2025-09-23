from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

from narwhals._plan.common import flatten_hash_safe
from narwhals._plan.typing import NativeDataFrameT, NativeFrameT, NativeSeriesT, Seq
from narwhals._typing_compat import TypeVar
from narwhals._utils import Version
from narwhals.exceptions import ComputeError

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias, TypeIs

    from narwhals._plan import expressions as ir
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
    from narwhals._plan.expressions.ranges import IntRange
    from narwhals._plan.expressions.strings import ConcatStr
    from narwhals._plan.options import SortMultipleOptions
    from narwhals._plan.series import Series
    from narwhals._plan.typing import OneOrIterable
    from narwhals.dtypes import DType
    from narwhals.typing import (
        ConcatMethod,
        Into1DArray,
        IntoDType,
        IntoSchema,
        NonNestedLiteral,
        PythonLiteral,
        _1DArray,
    )

T = TypeVar("T")
R_co = TypeVar("R_co", covariant=True)
LengthT = TypeVar("LengthT")
NativeT_co = TypeVar("NativeT_co", covariant=True, default=Any)

ConcatT1 = TypeVar("ConcatT1")
ConcatT2 = TypeVar("ConcatT2", default=ConcatT1)

ColumnT = TypeVar("ColumnT")
ColumnT_co = TypeVar("ColumnT_co", covariant=True)

ExprAny: TypeAlias = "CompliantExpr[Any, Any]"
ScalarAny: TypeAlias = "CompliantScalar[Any, Any]"
SeriesAny: TypeAlias = "CompliantSeries[Any]"
FrameAny: TypeAlias = "CompliantBaseFrame[Any, Any]"
DataFrameAny: TypeAlias = "CompliantDataFrame[Any, Any, Any]"
NamespaceAny: TypeAlias = "CompliantNamespace[Any, Any, Any]"

EagerExprAny: TypeAlias = "EagerExpr[Any, Any]"
EagerScalarAny: TypeAlias = "EagerScalar[Any, Any]"
EagerDataFrameAny: TypeAlias = "EagerDataFrame[Any, Any, Any]"

LazyExprAny: TypeAlias = "LazyExpr[Any, Any, Any]"
LazyScalarAny: TypeAlias = "LazyScalar[Any, Any, Any]"

ExprT_co = TypeVar("ExprT_co", bound=ExprAny, covariant=True)
ScalarT = TypeVar("ScalarT", bound=ScalarAny)
ScalarT_co = TypeVar("ScalarT_co", bound=ScalarAny, covariant=True)
SeriesT = TypeVar("SeriesT", bound=SeriesAny)
SeriesT_co = TypeVar("SeriesT_co", bound=SeriesAny, covariant=True)
FrameT = TypeVar("FrameT", bound=FrameAny)
FrameT_co = TypeVar("FrameT_co", bound=FrameAny, covariant=True)
FrameT_contra = TypeVar("FrameT_contra", bound=FrameAny, contravariant=True)
DataFrameT = TypeVar("DataFrameT", bound=DataFrameAny)
NamespaceT_co = TypeVar("NamespaceT_co", bound="NamespaceAny", covariant=True)

EagerExprT_co = TypeVar("EagerExprT_co", bound=EagerExprAny, covariant=True)
EagerScalarT_co = TypeVar("EagerScalarT_co", bound=EagerScalarAny, covariant=True)
EagerDataFrameT = TypeVar("EagerDataFrameT", bound=EagerDataFrameAny)

LazyExprT_co = TypeVar("LazyExprT_co", bound=LazyExprAny, covariant=True)
LazyScalarT_co = TypeVar("LazyScalarT_co", bound=LazyScalarAny, covariant=True)

Ctx: TypeAlias = "ExprDispatch[FrameT_contra, R_co, NamespaceAny]"
"""Type of an unknown expression dispatch context.

- `FrameT_contra`: Compliant data/lazyframe
- `R_co`: Upper bound return type of the context
"""


class SupportsNarwhalsNamespace(Protocol[NamespaceT_co]):
    def __narwhals_namespace__(self) -> NamespaceT_co: ...


def namespace(obj: SupportsNarwhalsNamespace[NamespaceT_co], /) -> NamespaceT_co:
    """Return the compliant namespace."""
    return obj.__narwhals_namespace__()


# NOTE: Unlike the version in `nw._utils`, here `.version` it is public
class StoresVersion(Protocol):
    _version: Version

    @property
    def version(self) -> Version:
        """Narwhals API version (V1 or MAIN)."""
        return self._version


class SupportsBroadcast(Protocol[SeriesT, LengthT]):
    """Minimal broadcasting for `Expr` results."""

    @classmethod
    def from_series(cls, series: SeriesT, /) -> Self: ...
    def to_series(self) -> SeriesT: ...
    def broadcast(self, length: LengthT, /) -> SeriesT: ...
    def _length(self) -> LengthT:
        """Return the length of the current expression."""
        ...

    @classmethod
    def _length_max(cls, lengths: Sequence[LengthT], /) -> LengthT:
        """Return the maximum length among `exprs`."""
        ...

    @classmethod
    def _length_required(
        cls, exprs: Sequence[SupportsBroadcast[SeriesT, LengthT]], /
    ) -> LengthT | None:
        """Return the broadcast length, if all lengths do not equal the maximum."""

    @classmethod
    def _length_all(
        cls, exprs: Sequence[SupportsBroadcast[SeriesT, LengthT]], /
    ) -> Sequence[LengthT]:
        return [e._length() for e in exprs]

    @classmethod
    def align(
        cls, *exprs: OneOrIterable[SupportsBroadcast[SeriesT, LengthT]]
    ) -> Iterator[SeriesT]:
        exprs = tuple[SupportsBroadcast[SeriesT, LengthT], ...](flatten_hash_safe(exprs))
        length = cls._length_required(exprs)
        if length is None:
            for e in exprs:
                yield e.to_series()
        else:
            for e in exprs:
                yield e.broadcast(length)


class EagerBroadcast(Sized, SupportsBroadcast[SeriesT, int], Protocol[SeriesT]):
    """Determines expression length via the size of the container."""

    def _length(self) -> int:
        return len(self)

    @classmethod
    def _length_max(cls, lengths: Sequence[int], /) -> int:
        return max(lengths)

    @classmethod
    def _length_required(
        cls, exprs: Sequence[SupportsBroadcast[SeriesT, int]], /
    ) -> int | None:
        lengths = cls._length_all(exprs)
        max_length = cls._length_max(lengths)
        required = any(len_ != max_length for len_ in lengths)
        return max_length if required else None


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


class CompliantNamespace(StoresVersion, Protocol[FrameT, ExprT_co, ScalarT_co]):
    @property
    def _frame(self) -> type[FrameT]: ...
    @property
    def _expr(self) -> type[ExprT_co]: ...
    @property
    def _scalar(self) -> type[ScalarT_co]: ...
    def col(self, node: ir.Column, frame: FrameT, name: str) -> ExprT_co: ...
    def lit(
        self, node: ir.Literal[Any], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def len(self, node: ir.Len, frame: FrameT, name: str) -> ScalarT_co: ...
    def any_horizontal(
        self, node: FunctionExpr[boolean.AnyHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def all_horizontal(
        self, node: FunctionExpr[boolean.AllHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def sum_horizontal(
        self, node: FunctionExpr[F.SumHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def min_horizontal(
        self, node: FunctionExpr[F.MinHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def max_horizontal(
        self, node: FunctionExpr[F.MaxHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def mean_horizontal(
        self, node: FunctionExpr[F.MeanHorizontal], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def concat_str(
        self, node: FunctionExpr[ConcatStr], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def int_range(
        self, node: ir.RangeExpr[IntRange], frame: FrameT, name: str
    ) -> ExprT_co: ...


class EagerNamespace(
    EagerConcat[EagerDataFrameT, SeriesT],
    CompliantNamespace[EagerDataFrameT, EagerExprT_co, EagerScalarT_co],
    Protocol[EagerDataFrameT, SeriesT, EagerExprT_co, EagerScalarT_co],
):
    @property
    def _series(self) -> type[SeriesT]: ...
    @property
    def _dataframe(self) -> type[EagerDataFrameT]: ...
    @property
    def _frame(self) -> type[EagerDataFrameT]:
        return self._dataframe

    def _is_series(self, obj: Any) -> TypeIs[SeriesT]:
        return isinstance(obj, self._series)

    def _is_dataframe(self, obj: Any) -> TypeIs[EagerDataFrameT]:
        return isinstance(obj, self._dataframe)

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
    def len(self, node: ir.Len, frame: EagerDataFrameT, name: str) -> EagerScalarT_co:
        return self._scalar.from_python(
            len(frame), name or node.name, dtype=None, version=frame.version
        )


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
    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], /, *, schema: IntoSchema | None = None
    ) -> Self: ...
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


class CompliantGroupBy(Protocol[FrameT_co]):
    @property
    def compliant(self) -> FrameT_co: ...
    def agg(self, *args: Any, **kwds: Any) -> FrameT_co: ...


class DataFrameGroupBy(CompliantGroupBy[DataFrameT], Protocol[DataFrameT]):
    _keys: Seq[NamedIR]
    _keys_names: Seq[str]

    @classmethod
    def by_names(
        cls, df: DataFrameT, names: Seq[str], /, *, drop_null_keys: bool = False
    ) -> DataFrameGroupBy[DataFrameT]: ...
    @classmethod
    def by_named_irs(
        cls, df: DataFrameT, irs: Seq[NamedIR], /
    ) -> DataFrameGroupBy[DataFrameT]: ...
    def __iter__(self) -> Iterator[tuple[Any, DataFrameT]]: ...
    @property
    def keys(self) -> Seq[NamedIR]:
        return self._keys

    @property
    def keys_names(self) -> Seq[str]:
        if names := self._keys_names:
            return names
        if keys := self.keys:
            return tuple(e.name for e in keys)
        msg = "at least one key is required in a group_by operation"
        raise ComputeError(msg)

    def agg(self, irs: Seq[NamedIR]) -> DataFrameT: ...


class EagerDataFrame(
    CompliantDataFrame[SeriesT, NativeDataFrameT, NativeSeriesT],
    Protocol[SeriesT, NativeDataFrameT, NativeSeriesT],
):
    def __narwhals_namespace__(self) -> EagerNamespace[Self, SeriesT, Any, Any]: ...
    def select(self, irs: Seq[NamedIR]) -> Self:
        return self.__narwhals_namespace__()._concat_horizontal(self._evaluate_irs(irs))

    def with_columns(self, irs: Seq[NamedIR]) -> Self:
        return self.__narwhals_namespace__()._concat_horizontal(self._evaluate_irs(irs))


class CompliantSeries(StoresVersion, Protocol[NativeSeriesT]):
    _native: NativeSeriesT
    _name: str

    def __narwhals_series__(self) -> Self:
        return self

    @property
    def native(self) -> NativeSeriesT:
        return self._native

    @property
    def dtype(self) -> DType: ...
    @property
    def name(self) -> str:
        return self._name

    def to_narwhals(self) -> Series[NativeSeriesT]:
        from narwhals._plan.series import Series

        return Series[NativeSeriesT]._from_compliant(self)

    @classmethod
    def from_native(
        cls, native: NativeSeriesT, name: str = "", /, *, version: Version = Version.MAIN
    ) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        obj._name = name
        obj._version = version
        return obj

    @classmethod
    def from_numpy(
        cls, data: Into1DArray, name: str = "", /, *, version: Version = Version.MAIN
    ) -> Self: ...
    @classmethod
    def from_iterable(
        cls,
        data: Iterable[Any],
        *,
        version: Version,
        name: str = "",
        dtype: IntoDType | None = None,
    ) -> Self: ...
    def _with_native(self, native: NativeSeriesT) -> Self:
        return self.from_native(native, self.name, version=self.version)

    def alias(self, name: str) -> Self:
        return self.from_native(self.native, name, version=self.version)

    def cast(self, dtype: IntoDType) -> Self: ...
    def __len__(self) -> int:
        return len(self.native)

    def to_list(self) -> list[Any]: ...
    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _1DArray: ...
