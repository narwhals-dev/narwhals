from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from narwhals._plan import aggregation as agg, expr
from narwhals._plan.common import ExprIR, NamedIR, flatten_hash_safe
from narwhals._typing_compat import TypeVar
from narwhals._utils import Version

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.dummy import DummySeries
    from narwhals.typing import IntoDType, NonNestedLiteral, PythonLiteral

T = TypeVar("T")
SeriesT = TypeVar("SeriesT")
SeriesT_co = TypeVar("SeriesT_co", covariant=True)
FrameT = TypeVar("FrameT")
FrameT_co = TypeVar("FrameT_co", covariant=True)
FrameT_contra = TypeVar("FrameT_contra", contravariant=True)
OneOrIterable: TypeAlias = "T | Iterable[T]"
LengthT = TypeVar("LengthT")
NativeT_co = TypeVar("NativeT_co", covariant=True, default=Any)
ExprAny: TypeAlias = "CompliantExpr[Any, Any]"
ScalarAny: TypeAlias = "CompliantScalar[Any, Any]"
ExprT_co = TypeVar("ExprT_co", bound=ExprAny, covariant=True)
ScalarT = TypeVar("ScalarT", bound="CompliantScalar[Any, Any]")
ScalarT_co = TypeVar("ScalarT_co", bound="CompliantScalar[Any, Any]", covariant=True)


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

    @property
    def version(self) -> Version: ...
    @property
    def name(self) -> str: ...

    @classmethod
    def from_native(
        cls, native: Any, name: str = "", /, version: Version = Version.MAIN
    ) -> Self: ...

    def _with_native(self, native: Any, name: str = "", /) -> Self:
        return self.from_native(native, name or self.name, self.version)

    # entry points
    @classmethod
    def col(cls, node: expr.Column, frame: FrameT_contra, name: str) -> Self: ...

    @classmethod
    def lit(
        cls,
        node: expr.Literal[NonNestedLiteral] | expr.Literal[DummySeries[Any]],
        frame: FrameT_contra,
        name: str,
    ) -> CompliantScalar[FrameT_contra, SeriesT_co] | Self: ...

    # series & scalar
    def cast(self, node: expr.Cast, frame: FrameT_contra, name: str) -> Self: ...
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

    _DISPATCH: ClassVar[Mapping[type[ExprIR], Callable[..., ExprAny]]] = {
        expr.Column: col,
        expr.Literal: lit,
        expr.Cast: cast,
        expr.Sort: sort,
        expr.SortBy: sort_by,
        expr.Filter: filter,
        agg.First: first,
        agg.Last: last,
        agg.ArgMin: arg_min,
        agg.ArgMax: arg_max,
        agg.Sum: sum,
        agg.NUnique: n_unique,
        agg.Std: std,
        agg.Var: var,
        agg.Quantile: quantile,
        agg.Count: count,
        agg.Max: max,
        agg.Mean: mean,
        agg.Median: median,
        agg.Min: min,
    }

    def _dispatch(self, named_ir: NamedIR[ExprIR], frame: FrameT_contra) -> ExprAny:
        return self._dispatch_inner(named_ir.expr, frame, named_ir.name)

    def _dispatch_inner(self, node: ExprIR, frame: FrameT_contra, name: str) -> ExprAny:
        method = self._DISPATCH[node.__class__]
        return method(self, node, frame, name)


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

    def _with_evaluated(self, evaluated: Any, name: str = "") -> Self:
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
        return self._with_evaluated(self._evaluated)

    def sort_by(self, node: expr.SortBy, frame: FrameT_contra, name: str) -> Self:
        return self._with_evaluated(self._evaluated)

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


class CompliantNamespace(Protocol[FrameT_co, SeriesT_co, ExprT_co, ScalarT_co]):
    """Need to hold `Expr` and `Scalar` types outside of their defs.

    Likely, re-wrapping the output types will work like:


        ns = DataFrame().__narwhals_namespace__()
        if ns._expr.is_native(out):
            ns._expr.from_native(out, ...)
        elif ns._scalar.is_native(out):
            ns._scalar.from_native(out, ...)
        else:
            assert_never(out)

    Currently that is causing issues in `ArrowExpr2._with_native`
    """

    @property
    def _expr(self) -> type[ExprT_co]: ...
    @property
    def _scalar(self) -> type[ScalarT_co]: ...
    @property
    def _series(self) -> type[SeriesT_co]: ...
    @property
    def _dataframe(self) -> type[FrameT_co]: ...
