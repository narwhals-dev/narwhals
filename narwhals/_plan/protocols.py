from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, overload

from narwhals._plan import aggregation as agg, expr
from narwhals._plan.common import ExprIR, NamedIR, flatten_hash_safe
from narwhals._plan.typing import NativeFrameT, NativeSeriesT, Seq
from narwhals._typing_compat import TypeVar
from narwhals._utils import Version, _hasattr_static

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.dummy import DummyFrame, DummySeries
    from narwhals._plan.options import SortMultipleOptions
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType
    from narwhals.schema import Schema
    from narwhals.typing import IntoDType, NonNestedLiteral, PythonLiteral

T = TypeVar("T")
R_co = TypeVar("R_co", covariant=True)
OneOrIterable: TypeAlias = "T | Iterable[T]"
LengthT = TypeVar("LengthT")
NativeT_co = TypeVar("NativeT_co", covariant=True, default=Any)

ExprAny: TypeAlias = "CompliantExpr[Any, Any]"
ScalarAny: TypeAlias = "CompliantScalar[Any, Any]"
SeriesAny: TypeAlias = "DummyCompliantSeries[Any]"
FrameAny: TypeAlias = "DummyCompliantFrame[Any, Any, Any]"
NamespaceAny: TypeAlias = "CompliantNamespace[Any, Any, Any, Any]"

ExprT_co = TypeVar("ExprT_co", bound=ExprAny, covariant=True)
ScalarT = TypeVar("ScalarT", bound=ScalarAny)
ScalarT_co = TypeVar("ScalarT_co", bound=ScalarAny, covariant=True)
SeriesT = TypeVar("SeriesT", bound=SeriesAny)
SeriesT_co = TypeVar("SeriesT_co", bound=SeriesAny, covariant=True)
FrameT = TypeVar("FrameT", bound=FrameAny)
FrameT_contra = TypeVar("FrameT_contra", bound=FrameAny, contravariant=True)
NamespaceT_co = TypeVar("NamespaceT_co", bound="NamespaceAny", covariant=True)

EagerExprT_co = TypeVar("EagerExprT_co", bound="EagerExpr[Any, Any]", covariant=True)
EagerScalarT_co = TypeVar(
    "EagerScalarT_co", bound="EagerScalar[Any, Any]", covariant=True
)


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


class CompliantExpr(Protocol[FrameT_contra, SeriesT_co]):
    """Getting a bit tricky, just storing notes.

    - Separating series/scalar makes a lot of sense
    - Handling the recursive case *without* intermediate (non-pyarrow) objects seems unachievable
      - Everywhere would need to first check if it a scalar, which isn't ergonomic
    - Broadcasting being separated is working
    - A lot of `pyarrow.compute` (section 2) can work on either scalar or series (`FunctionExpr`)
      - Aggregation can't, but that is already handled in `ExprIR`
      - `polars` noops on aggregating a scalar, which we might be able to support this way
    """

    _evaluated: Any
    """Compliant or native value."""

    _version: Version

    @property
    def version(self) -> Version:
        return self._version

    @property
    def name(self) -> str: ...

    @classmethod
    def from_native(
        cls, native: Any, name: str = "", /, version: Version = Version.MAIN
    ) -> Self: ...

    def _with_native(self, native: Any, name: str = "", /) -> Self:
        return self.from_native(native, name or self.name, self.version)

    # series & scalar
    def cast(self, node: expr.Cast, frame: FrameT_contra, name: str) -> Self: ...
    def binary_expr(
        self, node: expr.BinaryExpr, frame: FrameT_contra, name: str
    ) -> Self: ...
    def ternary_expr(
        self, node: expr.Ternary, frame: FrameT_contra, name: str
    ) -> Self: ...
    def over(self, node: expr.WindowExpr, frame: FrameT_contra, name: str) -> Self: ...
    def over_ordered(
        self, node: expr.OrderedWindowExpr, frame: FrameT_contra, name: str
    ) -> Self: ...
    def map_batches(
        self, node: expr.AnonymousExpr, frame: FrameT_contra, name: str
    ) -> Self: ...
    def rolling_expr(
        self, node: expr.RollingExpr, frame: FrameT_contra, name: str
    ) -> Self: ...
    def function_expr(
        self, node: expr.FunctionExpr, frame: FrameT_contra, name: str
    ) -> Self: ...
    # series only (section 3)
    def sort(self, node: expr.Sort, frame: FrameT_contra, name: str) -> Self: ...
    def sort_by(self, node: expr.SortBy, frame: FrameT_contra, name: str) -> Self: ...
    def filter(self, node: expr.Filter, frame: FrameT_contra, name: str) -> Self: ...
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


class ExprDispatch(Protocol[FrameT_contra, R_co, NamespaceT_co]):
    _DISPATCH: ClassVar[Mapping[type[ExprIR], Callable[[Any, ExprIR, Any, str], Any]]] = {
        expr.Column: lambda self, node, frame, name: self.__narwhals_namespace__().col(
            node, frame, name
        ),
        expr.Literal: lambda self, node, frame, name: self.__narwhals_namespace__().lit(
            node, frame, name
        ),
        expr.Len: lambda self, node, frame, name: self.__narwhals_namespace__().len(
            node, frame, name
        ),
        expr.Cast: lambda self, node, frame, name: self.cast(node, frame, name),
        expr.Sort: lambda self, node, frame, name: self.sort(node, frame, name),
        expr.SortBy: lambda self, node, frame, name: self.sort_by(node, frame, name),
        expr.Filter: lambda self, node, frame, name: self.filter(node, frame, name),
        agg.First: lambda self, node, frame, name: self.first(node, frame, name),
        agg.Last: lambda self, node, frame, name: self.last(node, frame, name),
        agg.ArgMin: lambda self, node, frame, name: self.arg_min(node, frame, name),
        agg.ArgMax: lambda self, node, frame, name: self.arg_max(node, frame, name),
        agg.Sum: lambda self, node, frame, name: self.sum(node, frame, name),
        agg.NUnique: lambda self, node, frame, name: self.n_unique(node, frame, name),
        agg.Std: lambda self, node, frame, name: self.std(node, frame, name),
        agg.Var: lambda self, node, frame, name: self.var(node, frame, name),
        agg.Quantile: lambda self, node, frame, name: self.quantile(node, frame, name),
        agg.Count: lambda self, node, frame, name: self.count(node, frame, name),
        agg.Max: lambda self, node, frame, name: self.max(node, frame, name),
        agg.Mean: lambda self, node, frame, name: self.mean(node, frame, name),
        agg.Median: lambda self, node, frame, name: self.median(node, frame, name),
        agg.Min: lambda self, node, frame, name: self.min(node, frame, name),
        expr.BinaryExpr: lambda self, node, frame, name: self.binary_expr(
            node, frame, name
        ),
        expr.RollingExpr: lambda self, node, frame, name: self.rolling_expr(
            node, frame, name
        ),
        expr.AnonymousExpr: lambda self, node, frame, name: self.map_batches(
            node, frame, name
        ),
        expr.FunctionExpr: lambda self, node, frame, name: self.function_expr(
            node, frame, name
        ),
        expr.OrderedWindowExpr: lambda self, node, frame, name: self.over_ordered(
            node, frame, name
        ),
        expr.WindowExpr: lambda self, node, frame, name: self.over(node, frame, name),
        expr.Ternary: lambda self, node, frame, name: self.ternary_expr(
            node, frame, name
        ),
    }
    _version: Version

    def _dispatch(self, node: ExprIR, frame: FrameT_contra, name: str) -> R_co:
        if (method := self._DISPATCH.get(node.__class__)) and (
            result := method(self, node, frame, name)
        ):
            return result  # type: ignore[no-any-return]
        msg = f"Support for {node.__class__.__name__!r} is not yet implemented, got:\n{node!r}"
        raise NotImplementedError(msg)

    @classmethod
    def from_ir(cls, node: ExprIR, frame: FrameT_contra, name: str) -> R_co:
        obj = cls.__new__(cls)
        obj._version = frame.version
        return obj._dispatch(node, frame, name)

    @classmethod
    def from_named_ir(cls, named_ir: NamedIR[ExprIR], frame: FrameT_contra) -> R_co:
        return cls.from_ir(named_ir.expr, frame, named_ir.name)

    # NOTE: Needs to stay `covariant` and never be used as a parameter
    def __narwhals_namespace__(self) -> NamespaceT_co: ...


class CompliantScalar(
    CompliantExpr[FrameT_contra, SeriesT_co], Protocol[FrameT_contra, SeriesT_co]
):
    _name: str
    _version: Version

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> Version:
        return self._version

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

    def _cast_float(self, node: ExprIR, frame: FrameT_contra, name: str) -> Self:
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

    def sort(self, node: expr.Sort, frame: FrameT_contra, name: str) -> Self:
        return self._with_evaluated(self._evaluated, name)

    def sort_by(self, node: expr.SortBy, frame: FrameT_contra, name: str) -> Self:
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


class LazyScalar(
    CompliantScalar[FrameT_contra, SeriesT],
    LazyExpr[FrameT_contra, SeriesT, LengthT],
    Protocol[FrameT_contra, SeriesT, LengthT],
): ...


class CompliantNamespace(Protocol[FrameT, SeriesT_co, ExprT_co, ScalarT_co]):
    """Need to hold `Expr` and `Scalar` types outside of their defs.

    Likely, re-wrapping the output types will work like:


        ns = DataFrame().__narwhals_namespace__()
        if ns._expr.is_native(out):
            ns._expr.from_native(out, ...)
        elif ns._scalar.is_native(out):
            ns._scalar.from_native(out, ...)
        else:
            assert_never(out)
    """

    _version: Version

    @property
    def _dataframe(self) -> type[FrameT]: ...
    @property
    def _series(self) -> type[SeriesT_co]: ...
    @property
    def _expr(self) -> type[ExprT_co]: ...
    @property
    def _scalar(self) -> type[ScalarT_co]: ...

    @property
    def version(self) -> Version:
        return self._version

    def col(self, node: expr.Column, frame: FrameT, name: str) -> ExprT_co: ...
    def lit(
        self, node: expr.Literal[Any], frame: FrameT, name: str
    ) -> ExprT_co | ScalarT_co: ...
    def len(self, node: expr.Len, frame: FrameT, name: str) -> ScalarT_co: ...


class EagerNamespace(
    CompliantNamespace[FrameT, SeriesT_co, EagerExprT_co, EagerScalarT_co],
    Protocol[FrameT, SeriesT_co, EagerExprT_co, EagerScalarT_co],
):
    @overload
    def lit(
        self, node: expr.Literal[NonNestedLiteral], frame: FrameT, name: str
    ) -> EagerScalarT_co: ...
    @overload
    def lit(
        self, node: expr.Literal[DummySeries[Any]], frame: FrameT, name: str
    ) -> EagerExprT_co: ...
    @overload
    def lit(
        self,
        node: expr.Literal[NonNestedLiteral] | expr.Literal[DummySeries[Any]],
        frame: FrameT,
        name: str,
    ) -> EagerExprT_co | EagerScalarT_co: ...
    def lit(
        self, node: expr.Literal[Any], frame: FrameT, name: str
    ) -> EagerExprT_co | EagerScalarT_co: ...

    def len(self, node: expr.Len, frame: FrameT, name: str) -> EagerScalarT_co:
        return self._scalar.from_python(
            len(frame), name or node.name, dtype=None, version=frame.version
        )


class DummyCompliantFrame(Protocol[SeriesT, NativeFrameT, NativeSeriesT]):
    _native: NativeFrameT
    _version: Version

    def __narwhals_namespace__(self) -> Any: ...

    @property
    def native(self) -> NativeFrameT:
        return self._native

    @property
    def version(self) -> Version:
        return self._version

    @property
    def columns(self) -> list[str]: ...

    def to_narwhals(self) -> DummyFrame[NativeFrameT, NativeSeriesT]: ...

    @classmethod
    def from_native(cls, native: NativeFrameT, /, version: Version) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        obj._version = version
        return obj

    def _with_native(self, native: NativeFrameT) -> Self:
        return self.from_native(native, self.version)

    @classmethod
    def from_series(
        cls, series: Iterable[SeriesT] | SeriesT, *more_series: SeriesT
    ) -> Self:
        """Return a new DataFrame, horizontally concatenating multiple Series."""
        ...

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        /,
        *,
        schema: Mapping[str, DType] | Schema | None = None,
    ) -> Self: ...

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

    @property
    def schema(self) -> Mapping[str, DType]: ...

    def _evaluate_irs(self, nodes: Iterable[NamedIR[ExprIR]], /) -> Iterator[SeriesT]: ...

    def select(self, irs: Seq[NamedIR], projected: FrozenSchema) -> Self:
        return self.from_series(self._evaluate_irs(irs))

    def sort(
        self, by: Seq[NamedIR], options: SortMultipleOptions, projected: FrozenSchema
    ) -> Self: ...


class DummyCompliantSeries(Protocol[NativeSeriesT]):
    _native: NativeSeriesT
    _name: str
    _version: Version

    def __narwhals_series__(self) -> Self:
        return self

    @property
    def native(self) -> NativeSeriesT:
        return self._native

    @property
    def version(self) -> Version:
        return self._version

    @property
    def dtype(self) -> DType: ...

    @property
    def name(self) -> str:
        return self._name

    def to_narwhals(self) -> DummySeries[NativeSeriesT]:
        from narwhals._plan.dummy import DummySeries

        return DummySeries[NativeSeriesT]._from_compliant(self)

    @classmethod
    def from_native(
        cls, native: NativeSeriesT, name: str = "", /, *, version: Version = Version.MAIN
    ) -> Self:
        name = name or (
            getattr(native, "name", name) if _hasattr_static(native, "name") else name
        )
        obj = cls.__new__(cls)
        obj._native = native
        obj._name = name
        obj._version = version
        return obj

    def _with_native(self, native: NativeSeriesT) -> Self:
        return self.from_native(native, self.name, version=self.version)

    def alias(self, name: str) -> Self:
        return self.from_native(self.native, name, version=self.version)

    def __len__(self) -> int:
        return len(self.native)

    def to_list(self) -> list[Any]: ...
