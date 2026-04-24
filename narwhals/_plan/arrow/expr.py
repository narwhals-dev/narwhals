from __future__ import annotations

from collections.abc import Callable, Iterable
from types import MethodType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, final, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan import common, expressions as ir
from narwhals._plan._function import UnaryFunction as _UnaryFunction
from narwhals._plan._guards import (
    is_function_expr,
    is_iterable_reject,
    is_python_literal,
    is_seq_column,
)
from narwhals._plan._namespace import namespace
from narwhals._plan.arrow import functions as fn, group_by
from narwhals._plan.arrow.namespace import ArrowNamespace as Namespace
from narwhals._plan.arrow.series import ArrowSeries as Series
from narwhals._plan.arrow.typing import ChunkedArrayAny, Native, ScalarAny as NativeScalar
from narwhals._plan.common import temp
from narwhals._plan.compliant import typing as ct
from narwhals._plan.compliant.accessors import (
    ExprCatNamespace,
    ExprListNamespace,
    ExprStringNamespace,
    ExprStructNamespace,
)
from narwhals._plan.compliant.expr import EagerExpr
from narwhals._plan.compliant.scalar import EagerScalar
from narwhals._plan.exceptions import shape_error
from narwhals._plan.expressions import FunctionExpr as FExpr, functions as F
from narwhals._plan.expressions.boolean import (
    IsDuplicated,
    IsFirstDistinct,
    IsInExpr,
    IsInSeq,
    IsInSeries,
    IsLastDistinct,
    IsUnique,
)
from narwhals._typing_compat import TypeVar
from narwhals._utils import Version, not_implemented, qualified_type_name
from narwhals.exceptions import InvalidOperationError, ShapeError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.arrow.dataframe import ArrowDataFrame as Frame
    from narwhals._plan.expressions import lists, strings
    from narwhals._plan.expressions.aggregation import (
        ArgMax,
        ArgMin,
        Count,
        First,
        Last,
        Len,
        Max,
        Mean,
        Median,
        Min,
        NUnique,
        Quantile,
        Std,
        Sum,
        Var,
    )
    from narwhals._plan.expressions.boolean import IsBetween
    from narwhals._plan.expressions.struct import FieldByName
    from narwhals._plan.typing import Seq
    from narwhals.typing import IntoDType, PythonLiteral

    Expr: TypeAlias = "ArrowExpr"
    Scalar: TypeAlias = "ArrowScalar"

Incomplete: TypeAlias = Any

_Self: TypeAlias = "_ArrowDispatch[Any]"
"""The upper bound for `S*` type vars used with `_ArrowDispatch`.

Similar to `typing.Self`, but to refer to the type *outside* of the enclosing class.
This is needed for getting the descriptor/decorator stuff to work nicely with subclasses.
"""

_AccessorSelf: TypeAlias = "ArrowCatNamespace[Any] | ArrowListNamespace[Any] | ArrowStringNamespace[Any] | ArrowStructNamespace[Any]"
"""Equivalent to `_Self`, for namespaces."""

UnaryFn: TypeAlias = _UnaryFunction
"""The upper bound for `U*` type vars used with `_ArrowDispatch`.

Nothing to fancy here, just a short way to say *subclasses of `UnaryFunction`*.
"""


_U_contra = TypeVar("_U_contra", bound=UnaryFn, contravariant=True)
_NativeT = TypeVar("_NativeT", bound="Native", default="Native")


UnaryPartial: TypeAlias = Callable[[ct.Self_, _U_contra, _NativeT], _NativeT]
"""The type of the method being *decorated by* `@unary.partial`."""

FunctionImplMethod = ct.FunctionImplMethod[ct.Self_, ct.F_contra, "Frame", ct.R]
"""The type of the wrapper method *produced by* any `unary` constructor."""

BoundFunctionImplMethod = ct.BoundFunctionImplMethod[
    ct.F_contra, "Frame", "Expr | Scalar"
]

S1 = TypeVar("S1", bound=_Self)
"""`_ArrowDispatch` scoped to an instance of `unary`."""
S2 = TypeVar("S2", bound=_Self)
"""`_ArrowDispatch` scoped to the `@staticmethod`(s) of `unary`."""
U1 = TypeVar("U1", bound=UnaryFn, contravariant=True, default=UnaryFn)  # noqa: PLC0105
"""`UnaryFunction` scoped to an instance of `unary`."""
U2 = TypeVar("U2", bound=UnaryFn, contravariant=True, default=UnaryFn)  # noqa: PLC0105
"""`UnaryFunction` scoped to the `@staticmethod`(s) of `unary`."""

Native_co = TypeVar("Native_co", bound="Native", covariant=True, default="Native")
"""The return type of `_ArrowDispatch.native`."""

F_co = TypeVar("F_co", bound=Callable[..., Any], covariant=True)


class _BaseWrapper(Generic[F_co]):
    __slots__ = ("__func__",)
    __func__: F_co

    def __init__(self, f: F_co, /) -> None:
        self.__func__ = f

    def __get__(
        self, instance: Incomplete | None, owner: type[Any] | None = None, /
    ) -> Incomplete | Self:
        if instance is None:
            return self
        return self._bind(instance)

    def _bind(self, instance: Incomplete) -> Incomplete:
        return MethodType(self.__func__, instance)

    if TYPE_CHECKING:
        # NOTE: This smooths over a lot of really ugly parts
        def __call__(self, *args: Incomplete, **kwds: Incomplete) -> Incomplete:
            raise NotImplementedError


@final
class unary(_BaseWrapper[FunctionImplMethod[S1, U1, S1]], Generic[S1, U1]):
    """Factories for implementing functions that dispatch a single expression.

    Provides two ways to write an implementation which cares only about the native parts.

    The boring stuff either side is wrapped around the function (`no_args`) or method (`partial`).

    ## Notes
    Separate-scoped `TypeVar`s emulate PEP 695 syntax
    """

    __slots__ = ()

    @staticmethod
    def partial(fn_partial: UnaryPartial[S2, U2, Native_co], /) -> Callable[..., S2]:
        """Decorator to fill in the boilerplate when implementing a `UnaryFunction`.

        The method being decorated should look like this (*parameter names are flexible*):

            @unary.partial
            def round(self, function: F.Round, previous: Native) -> Native:
                # 1. Destructure `function` to get additional non-expression arguments
                # 2. (optional) Do something interesting with them
                # 3. Use `previous` (the native result of the last expression)
                #    and the other arguments to perform the native operation
                # 4. Return the native result
                ...

        Tip:
            Consider using `unary.no_args(<impl>)` if you don't need anything from `node.function`.
        """

        def _wrapper(self: S2, node: FExpr[U2], frame: Frame, name: str) -> S2:
            previous = node.dispatch_arg(self, frame, name).native  # type: ignore[arg-type]
            return self._with_native(fn_partial(self, node.function, previous), name)  # type: ignore[arg-type]

        return unary(_wrapper)

    @staticmethod
    @overload
    def no_args(
        fn_native: Callable[[Native], Native], /
    ) -> Callable[..., Incomplete]: ...
    @staticmethod
    @overload
    def no_args(
        fn_native: Callable[[ChunkedArrayAny], ChunkedArrayAny], /
    ) -> Callable[..., Expr]: ...
    @staticmethod
    @overload
    def no_args(
        fn_native: Callable[[ChunkedArrayAny], NativeScalar], /
    ) -> Callable[..., Scalar]: ...
    @staticmethod
    def no_args(
        fn_native: Callable[[Native], Native] | Callable[[ChunkedArrayAny], Native], /
    ) -> Callable[..., Incomplete]:
        """Non-decorating function wrapper to fill in the boilerplate when implementing a `UnaryFunction`.

        Use this when the only argument to the function is the result of the last expression:

            is_finite = unary.no_args(fn.is_finite)
        """

        def _wrapper(self: _Self, node: FExpr[UnaryFn], frame: Frame, name: str) -> _Self:
            previous = node.dispatch_arg(self, frame, name).native  # type: ignore[arg-type]
            return self._with_native(fn_native(previous), name)  # type: ignore[arg-type]

        return unary(_wrapper)


AS1 = TypeVar("AS1", bound=_AccessorSelf)
AS2 = TypeVar("AS2", bound=_AccessorSelf)

_ExprOrScalar: TypeAlias = "Expr | Scalar"
"""Forward ref safety for `unary_accessor`."""


@final
class unary_accessor(  # noqa: N801
    _BaseWrapper[FunctionImplMethod[AS1, U1, _ExprOrScalar]], Generic[AS1, U1]
):
    """`ArrowAccessor` equivalent of `unary`.

    Ideally, I want the end api to be something like:


        dispatch.unary.no_args(<function>)

        @dispatch.unary.partial
        def round(self: _ArrowDispatch, f: F.Round, previous: Native) -> Native: ...


        # Should be generic over `self.compliant`
        dispatch.accessor.unary.no_args(<accessor-function>)

        @dispatch.accessor.unary.partial
        def get(self: ArrowListNamespace, f: lists.Get, previous: Native) -> Native: ...
    """

    __slots__ = ()

    @staticmethod
    def partial(
        fn_partial: UnaryPartial[AS2, U2, Native], /
    ) -> Callable[..., Expr | Scalar]:
        def _wrapper(
            self: AS2, node: FExpr[U2], frame: Frame, name: str
        ) -> Expr | Scalar:
            compliant = self.compliant
            previous = node.dispatch_arg(compliant, frame, name).native
            result: Expr | Scalar = compliant._with_native(
                fn_partial(self, node.function, previous), name
            )
            return result

        return unary_accessor(_wrapper)

    @staticmethod
    @overload
    def no_args(
        fn_native: Callable[[ChunkedArrayAny], ChunkedArrayAny], /
    ) -> Callable[..., Expr]: ...
    @staticmethod
    @overload
    def no_args(
        fn_native: Callable[[ChunkedArrayAny], NativeScalar], /
    ) -> Callable[..., Scalar]: ...
    @staticmethod
    @overload
    def no_args(
        fn_native: Callable[[Native], Native], /
    ) -> Callable[..., Expr | Scalar]: ...
    @staticmethod
    def no_args(
        fn_native: Callable[[Native], Native] | Callable[[ChunkedArrayAny], Native], /
    ) -> Callable[..., Expr | Scalar]:
        def _wrapper(
            self: _AccessorSelf, node: FExpr[UnaryFn], frame: Frame, name: str
        ) -> Expr | Scalar:
            compliant = self.compliant
            previous = node.dispatch_arg(compliant, frame, name).native
            result: Expr | Scalar = compliant._with_native(fn_native(previous), name)
            return result

        return unary_accessor(_wrapper)


class _ArrowDispatch(EagerExpr["Frame", Series], Protocol[Native_co]):
    """Common to `Expr`, `Scalar` + their dependencies."""

    version: ClassVar[Version] = Version.MAIN

    @property
    def native(self) -> Native_co:
        raise NotImplementedError

    def _with_native(self, native: Any, name: str, /) -> Self:
        raise NotImplementedError

    def __narwhals_namespace__(self) -> Namespace:
        return Namespace()

    def cast(self, node: ir.Cast, frame: Frame, name: str, /) -> Self:
        data_type = fn.dtype_native(node.dtype, self.version)
        native = node.expr.dispatch(self, frame, name).native  # type: ignore[arg-type]
        return self._with_native(fn.cast(native, data_type), name)

    def pow(self, node: FExpr[F.Pow], frame: Frame, name: str, /) -> Self:
        base, exponent = node.dispatch_args(self, frame, name)  # type: ignore[arg-type]
        return self._with_native(fn.power(base.native, exponent.native), name)

    def fill_null(self, node: FExpr[F.FillNull], frame: Frame, name: str, /) -> Self:
        expr, value = node.dispatch_args(self, frame, name)  # type: ignore[arg-type]
        return self._with_native(pc.fill_null(expr.native, value.native), name)

    def is_between(self, node: FExpr[IsBetween], frame: Frame, name: str, /) -> Self:
        expr, lb, ub = node.dispatch_args(self, frame, name)  # type: ignore[arg-type]
        closed = node.function.closed
        result = fn.is_between(expr.native, lb.native, ub.native, closed=closed)
        return self._with_native(result, name)

    def is_in_expr(self, node: FExpr[IsInExpr], frame: Frame, name: str, /) -> Self:
        native, right = (s.native for s in node.dispatch_args(self, frame, name))  # type: ignore[arg-type]
        arr = fn.array(right) if isinstance(right, pa.Scalar) else right
        return self._with_native(fn.is_in(native, arr), name)

    @unary.partial
    def is_in_series(self, f: IsInSeries[ChunkedArrayAny], previous: Native) -> Native:
        return fn.is_in(previous, f.other.native)

    @unary.partial
    def is_in_seq(self, f: IsInSeq, previous: Native) -> Native:
        return fn.is_in(previous, fn.array(f.other))

    def binary_expr(self, node: ir.BinaryExpr, frame: Frame, name: str, /) -> Self:
        lhs, rhs = (
            node.left.dispatch(self, frame, name),  # type: ignore[arg-type]
            node.right.dispatch(self, frame, name),  # type: ignore[arg-type]
        )
        result = fn.binary(lhs.native, node.op.__class__, rhs.native)
        return self._with_native(result, name)

    def ternary_expr(self, node: ir.TernaryExpr, frame: Frame, name: str, /) -> Self:
        when = node.predicate.dispatch(self, frame, name)  # type: ignore[arg-type]
        then = node.truthy.dispatch(self, frame, name)  # type: ignore[arg-type]
        otherwise = node.falsy.dispatch(self, frame, name)  # type: ignore[arg-type]
        result = pc.if_else(when.native, then.native, otherwise.native)
        return self._with_native(result, name)

    @unary.partial
    def log(self, f: F.Log, previous: Native) -> Native:
        return fn.log(previous, f.base)

    @unary.partial
    def round(self, f: F.Round, previous: Native) -> Native:
        return fn.round(previous, f.decimals)

    def clip(self, node: FExpr[F.Clip], frame: Frame, name: str, /) -> Self:
        expr, lb, ub = node.dispatch_args(self, frame, name)  # type: ignore[arg-type]
        return self._with_native(fn.clip(expr.native, lb.native, ub.native), name)

    def clip_lower(self, node: FExpr[F.ClipLower], frame: Frame, name: str, /) -> Self:
        expr, other = node.dispatch_args(self, frame, name)  # type: ignore[arg-type]
        return self._with_native(fn.clip_lower(expr.native, other.native), name)

    def clip_upper(self, node: FExpr[F.ClipUpper], frame: Frame, name: str, /) -> Self:
        expr, other = node.dispatch_args(self, frame, name)  # type: ignore[arg-type]
        return self._with_native(fn.clip_upper(expr.native, other.native), name)

    @unary.partial
    def replace_strict(self, f: F.ReplaceStrict, previous: Native) -> Native:
        dtype = fn.dtype_native(f.return_dtype, self.version)
        return fn.replace_strict(previous, f.old, f.new, dtype)

    def replace_strict_default(
        self, node: FExpr[F.ReplaceStrictDefault], frame: Frame, name: str, /
    ) -> Self:
        f = node.function
        native, default = (s.native for s in node.dispatch_args(self, frame, name))  # type: ignore[arg-type]
        dtype = fn.dtype_native(f.return_dtype, self.version)
        result = fn.replace_strict_default(native, f.old, f.new, default, dtype)
        return self._with_native(result, name)

    not_: Callable[..., Self] = unary.no_args(fn.not_)
    is_finite: Callable[..., Self] = unary.no_args(fn.is_finite)
    is_nan: Callable[..., Self] = unary.no_args(fn.is_nan)
    is_null: Callable[..., Self] = unary.no_args(fn.is_null)
    is_not_nan: Callable[..., Self] = unary.no_args(fn.is_not_nan)
    is_not_null: Callable[..., Self] = unary.no_args(fn.is_not_null)
    exp: Callable[..., Self] = unary.no_args(fn.exp)
    sqrt: Callable[..., Self] = unary.no_args(fn.sqrt)
    ceil: Callable[..., Self] = unary.no_args(fn.ceil)
    floor: Callable[..., Self] = unary.no_args(fn.floor)
    abs: Callable[..., Self] = unary.no_args(fn.abs)
    any: Callable[..., Scalar] = unary.no_args(fn.any)
    all: Callable[..., Scalar] = unary.no_args(fn.all)


class ArrowExpr(_ArrowDispatch["ChunkedArrayAny"], EagerExpr["Frame", Series]):
    _evaluated: Series

    @property
    def name(self) -> str:
        return self._evaluated.name

    @classmethod
    def from_series(cls, series: Series, /) -> Self:
        obj = cls.__new__(cls)
        obj._evaluated = series
        return obj

    @classmethod
    def from_native(cls, native: ChunkedArrayAny, name: str = "", /) -> Self:
        return cls.from_series(Series.from_native(native, name))

    @overload
    def _with_native(self, result: ChunkedArrayAny, name: str, /) -> Self: ...
    @overload
    def _with_native(self, result: NativeScalar, name: str, /) -> Scalar: ...
    @overload
    def _with_native(self, result: Native, name: str, /) -> Scalar | Self: ...
    def _with_native(self, result: Native, name: str, /) -> Scalar | Self:
        if isinstance(result, pa.Scalar):
            return ArrowScalar.from_native(result, name)
        return self.from_native(result, name)

    def _dispatch_expr(self, node: ir.ExprIR, frame: Frame, name: str) -> Series:
        """Use instead of `_dispatch` *iff* an operation is only supported on `ChunkedArray`.

        There is no need to broadcast, as they may have a cheaper impl elsewhere (`CompliantScalar` or `ArrowScalar`).

        Mainly for the benefit of a type checker, but the equivalent `ArrowScalar._dispatch_expr` will raise if
        the assumption fails.
        """
        return node.dispatch(self, frame, name).to_series()

    @property
    def native(self) -> ChunkedArrayAny:
        return self._evaluated.native

    def to_series(self) -> Series:
        return self._evaluated

    # TODO @dangotbanned: Handle this `Series([...])` edge case higher up
    # Can occur from a len(1) series passed to `with_columns`, which becomes a literal
    def broadcast(self, length: int, /) -> Series:
        if (actual_len := len(self)) != length:
            if actual_len == 1:
                msg = (
                    f"Series {self.name}, length {actual_len} doesn't match the DataFrame height of {length}.\n\n"
                    "If you want an expression to be broadcasted, ensure it is a scalar (for instance by adding '.first()')."
                )
                raise ShapeError(msg)
            raise shape_error(length, actual_len)
        return self._evaluated

    def __len__(self) -> int:
        return len(self._evaluated)

    def sort(self, node: ir.Sort, frame: Frame, name: str, /) -> Expr:
        series = self._dispatch_expr(node.expr, frame, name)
        opts = node.options
        result = series.sort(descending=opts.descending, nulls_last=opts.nulls_last)
        return self.from_series(result)

    def sort_by(self, node: ir.SortBy, frame: Frame, name: str, /) -> Expr:
        if is_seq_column(node.by):
            # fastpath, roughly the same as `DataFrame.sort`, but only taking indices
            # of a single column
            keys: Sequence[str] = tuple(e.name for e in node.by)
            df = frame
        else:
            it_names = temp.column_names(frame)
            by = (self._dispatch_expr(e, frame, nm) for e, nm in zip(node.by, it_names))
            df = namespace(self).concat_series_horizontal(by)
            keys = df.columns
        indices = fn.sort_indices(df.native, *keys, options=node.options)
        series = self._dispatch_expr(node.expr, frame, name)
        return self.from_series(series.gather(indices))

    def filter(self, node: ir.Filter, frame: Frame, name: str, /) -> Expr:
        return self._with_native(
            self._dispatch_expr(node.expr, frame, name).native.filter(
                self._dispatch_expr(node.by, frame, name).native
            ),
            name,
        )

    def first(self, node: First, frame: Frame, name: str, /) -> Scalar:
        prev = self._dispatch_expr(node.expr, frame, name)
        native = prev.native
        result: NativeScalar = native[0] if len(prev) else fn.lit(None, native.type)
        return self._with_native(result, name)

    def last(self, node: Last, frame: Frame, name: str, /) -> Scalar:
        prev = self._dispatch_expr(node.expr, frame, name)
        native = prev.native
        result: NativeScalar = (
            native[len_ - 1] if (len_ := len(prev)) else fn.lit(None, native.type)
        )
        return self._with_native(result, name)

    def arg_min(self, node: ArgMin, frame: Frame, name: str, /) -> Scalar:
        native = self._dispatch_expr(node.expr, frame, name).native
        result = pc.index(native, fn.min(native))
        return self._with_native(result, name)

    def arg_max(self, node: ArgMax, frame: Frame, name: str, /) -> Scalar:
        native = self._dispatch_expr(node.expr, frame, name).native
        result: NativeScalar = pc.index(native, fn.max(native))
        return self._with_native(result, name)

    def sum(self, node: Sum, frame: Frame, name: str, /) -> Scalar:
        result = fn.sum(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def n_unique(self, node: NUnique, frame: Frame, name: str, /) -> Scalar:
        result = fn.n_unique(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def std(self, node: Std, frame: Frame, name: str, /) -> Scalar:
        result = fn.std(
            self._dispatch_expr(node.expr, frame, name).native, ddof=node.ddof
        )
        return self._with_native(result, name)

    def var(self, node: Var, frame: Frame, name: str, /) -> Scalar:
        result = fn.var(
            self._dispatch_expr(node.expr, frame, name).native, ddof=node.ddof
        )
        return self._with_native(result, name)

    def quantile(self, node: Quantile, frame: Frame, name: str, /) -> Scalar:
        result = fn.quantile(
            self._dispatch_expr(node.expr, frame, name).native,
            q=node.quantile,
            interpolation=node.interpolation,
        )[0]
        return self._with_native(result, name)

    def count(self, node: Count, frame: Frame, name: str, /) -> Scalar:
        result = fn.count(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def len(self, node: Len, frame: Frame, name: str, /) -> Scalar:
        result = fn.count(self._dispatch_expr(node.expr, frame, name).native, mode="all")
        return self._with_native(result, name)

    def max(self, node: Max, frame: Frame, name: str, /) -> Scalar:
        result: NativeScalar = fn.max(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def mean(self, node: Mean, frame: Frame, name: str, /) -> Scalar:
        result = fn.mean(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def median(self, node: Median, frame: Frame, name: str, /) -> Scalar:
        result = fn.median(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def min(self, node: Min, frame: Frame, name: str, /) -> Scalar:
        result: NativeScalar = fn.min(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def over(
        self,
        node: ir.Over,
        frame: Frame,
        name: str,
        *,
        sort_indices: pa.UInt64Array | None = None,
    ) -> Self:
        expr = node.expr
        by = node.partition_by
        if is_function_expr(expr) and isinstance(
            expr.function, (IsFirstDistinct, IsLastDistinct, IsUnique, IsDuplicated)
        ):
            return self._boolean_length_preserving(
                expr, frame, name, by, sort_indices=sort_indices
            )
        resolved = frame._grouper.by_irs(*by).agg_irs(expr.alias(name)).resolve(frame)
        results = frame.group_by_resolver(resolved).agg_over(resolved.aggs, sort_indices)
        return self.from_series(results.get_column(name))

    def over_ordered(
        self, node: ir.OverOrdered, frame: Frame, name: str, /
    ) -> Self | Scalar:
        by = node.order_by_names()
        indices = fn.sort_indices(frame.native, *by, options=node.sort_options)
        if node.partition_by:
            return self.over(node, frame, name, sort_indices=indices)
        evaluated = node.expr.dispatch(self, frame.gather(indices), name)
        if isinstance(evaluated, ArrowScalar):
            return evaluated
        return self.from_series(evaluated.broadcast(len(frame)).gather(indices))

    def _boolean_length_preserving(
        self,
        node: FExpr[IsFirstDistinct | IsLastDistinct | IsUnique | IsDuplicated],
        frame: Frame,
        name: str,
        partition_by: Seq[ir.ExprIR] = (),
        *,
        sort_indices: pa.UInt64Array | None = None,
    ) -> Self:
        # NOTE: This subset of functions can be expressed as a mask applied to indices
        into_agg, mask = group_by._BOOLEAN_LENGTH_PRESERVING[type(node.function)]
        idx_name = temp.column_name(frame)
        df = frame._with_columns([node.dispatch_arg(self, frame, name)])
        if sort_indices is not None:
            column = fn.unsort_indices(sort_indices)
            df = df._with_native(df.native.add_column(0, idx_name, column))
        else:
            df = df.with_row_index(idx_name)
        if not (partition_by or sort_indices is not None):
            aggregated = df.group_by_names((name,)).agg(
                (group_by.named_ir_agg(idx_name, into_agg),)
            )
        else:
            # TODO @dangotbanned: Try to align using `NamedIR` on both `agg` clauses
            agg_node = into_agg(expr=ir.col(idx_name))
            aggregated = df.group_by_agg_irs((ir.col(name), *partition_by), agg_node)
        index = df.to_series().alias(name)
        final_result = mask(index.native, aggregated.get_column(idx_name).native)
        return self.from_series(index._with_native(final_result))

    # NOTE: Can't implement in `EagerExpr` (like on `main`)
    # The version here is missing `__narwhals_namespace__`
    def map_batches(
        self, node: ir.AnonymousExpr, frame: Frame, name: str, /
    ) -> Self | Scalar:
        series = self._dispatch_expr(node.input[0], frame, name)
        udf = node.function.function
        udf_result: Series | Iterable[Any] | Any = udf(series)
        if node.is_scalar():
            return ArrowScalar.from_unknown(
                udf_result, name, dtype=node.function.return_dtype
            )
        if isinstance(udf_result, Series):
            result = udf_result
        elif isinstance(udf_result, Iterable) and not is_iterable_reject(udf_result):
            result = Series.from_iterable(udf_result, name=name)
        else:
            msg = (
                "`map_batches` with `returns_scalar=False` must return a Series; "
                f"found '{qualified_type_name(udf_result)}'.\n\nIf `returns_scalar` "
                "is set to `True`, a returned value can be a scalar value."
            )
            raise TypeError(msg)
        if dtype := node.function.return_dtype:
            result = result.cast(dtype)
        return self.from_series(result)

    @unary.partial
    def shift(self, f: F.Shift, previous: ChunkedArrayAny) -> ChunkedArrayAny:
        return fn.shift(previous, f.n)

    @unary.partial
    def rank(self, f: F.Rank, previous: ChunkedArrayAny) -> ChunkedArrayAny:
        return fn.rank(previous, f.options)

    def _cumulative(self, node: FExpr[F.CumAgg], frame: Frame, name: str, /) -> Self:
        native = node.dispatch_arg(self, frame, name).native
        return self._with_native(fn.cumulative(native, node.function), name)  # type: ignore[arg-type]

    def gather_every(
        self, node: FExpr[F.GatherEvery], frame: Frame, name: str, /
    ) -> Self:
        series = self._dispatch_expr(node.input[0], frame, name)
        n, offset = node.function.n, node.function.offset
        return self.from_series(series.gather_every(n=n, offset=offset))

    def sample_n(self, node: FExpr[F.SampleN], frame: Frame, name: str, /) -> Self:
        series = self._dispatch_expr(node.input[0], frame, name)
        func = node.function
        n, replace, seed = func.n, func.with_replacement, func.seed
        result = series.sample_n(n, with_replacement=replace, seed=seed)
        return self.from_series(result)

    def sample_frac(self, node: FExpr[F.SampleFrac], frame: Frame, name: str, /) -> Self:
        series = self._dispatch_expr(node.input[0], frame, name)
        func = node.function
        fraction, replace, seed = func.fraction, func.with_replacement, func.seed
        result = series.sample_frac(fraction, with_replacement=replace, seed=seed)
        return self.from_series(result)

    @unary.partial
    def fill_null_with_strategy(
        self, f: F.FillNullWithStrategy, previous: ChunkedArrayAny, /
    ) -> ChunkedArrayAny:
        return fn.fill_null_with_strategy(previous, f.strategy, f.limit)

    null_count = unary.no_args(fn.null_count)
    kurtosis = unary.no_args(fn.kurtosis)
    skew = unary.no_args(fn.skew)
    diff = unary.no_args(fn.diff)
    mode_all = unary.no_args(fn.mode_all)
    mode_any = unary.no_args(fn.mode_any)
    unique = unary.no_args(pc.unique)
    drop_nulls = unary.no_args(fn.drop_nulls)
    cum_count = cum_min = cum_max = cum_prod = cum_sum = _cumulative
    is_first_distinct = is_last_distinct = _boolean_length_preserving
    is_duplicated = is_unique = _boolean_length_preserving

    _ROLLING: ClassVar[Mapping[type[F.RollingWindow], Callable[..., Series]]] = {
        F.RollingSum: Series.rolling_sum,
        F.RollingMean: Series.rolling_mean,
        F.RollingVar: Series.rolling_var,
        F.RollingStd: Series.rolling_std,
    }

    def _rolling(self, node: FExpr[F.RollingWindow], frame: Frame, name: str, /) -> Self:
        s = self._dispatch_expr(node.input[0], frame, name)
        f = node.function
        return self.from_series(self._ROLLING[type(f)](s, **(f.options.to_dict())))

    rolling_sum = rolling_mean = rolling_std = rolling_var = _rolling

    # NOTE: Should not be returning a struct when all `include_*` are false
    # https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-ops/src/chunked_array/hist.rs#L223-L223
    def _hist_finish(self, data: Mapping[str, Any], name: str) -> Self:
        ns = namespace(self)
        if len(data) == 1:
            count = next(iter(data.values()))
            series = ns._series.from_iterable(count, name=name)
        else:
            series = ns._dataframe.from_dict(data).to_struct(name)
        return self.from_series(series)

    def hist_bins(self, node: FExpr[F.HistBins], frame: Frame, name: str, /) -> Self:
        native = self._dispatch_expr(node.input[0], frame, name).native
        func = node.function
        bins = func.bins
        include = func.include_breakpoint
        if len(bins) <= 1:
            data = func.empty_data
        elif fn.is_only_nulls(native, nan_is_null=True):
            data = fn.hist_zeroed_data(bins, include_breakpoint=include)
        else:
            data = fn.hist_bins(native, bins, include_breakpoint=include)
        return self._hist_finish(data, name)

    def hist_bin_count(
        self, node: FExpr[F.HistBinCount], frame: Frame, name: str, /
    ) -> Self:
        native = self._dispatch_expr(node.input[0], frame, name).native
        func = node.function
        bin_count = func.bin_count
        include = func.include_breakpoint
        if bin_count == 0:
            data = func.empty_data
        elif fn.is_only_nulls(native, nan_is_null=True):
            data = fn.hist_zeroed_data(bin_count, include_breakpoint=include)
        else:
            # NOTE: `Decimal` is not supported, but excluding it from the typing is surprisingly complicated
            # https://docs.rs/polars-core/0.52.0/polars_core/datatypes/enum.DataType.html#method.is_primitive_numeric
            lower: NativeScalar = fn.min(native)
            upper: NativeScalar = fn.max(native)
            if lower.equals(upper):
                # All data points are identical - use unit interval
                rhs = fn.lit(0.5)
                lower, upper = fn.sub(lower, rhs), fn.add(upper, rhs)
            bins = fn.linear_space(lower.as_py(), upper.as_py(), bin_count + 1)
            data = fn.hist_bins(native, bins, include_breakpoint=include)
        return self._hist_finish(data, name)

    @property
    def cat(self) -> ArrowCatNamespace[Expr]:
        return ArrowCatNamespace(self)

    @property
    def list(self) -> ArrowListNamespace[Expr]:
        return ArrowListNamespace(self)

    @property
    def str(self) -> ArrowStringNamespace[Expr]:
        return ArrowStringNamespace(self)

    @property
    def struct(self) -> ArrowStructNamespace[Expr]:
        return ArrowStructNamespace(self)

    dt = not_implemented()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    ewm_mean = not_implemented()


class ArrowScalar(_ArrowDispatch[NativeScalar], EagerScalar["Frame", Series]):
    _evaluated: NativeScalar
    _name: str

    @classmethod
    def from_native(cls, scalar: NativeScalar, name: str = "literal", /) -> Self:
        obj = cls.__new__(cls)
        obj._evaluated = scalar
        obj._name = name
        return obj

    @classmethod
    def from_python(
        cls,
        value: PythonLiteral,
        name: str = "literal",
        /,
        *,
        dtype: IntoDType | None = None,
    ) -> Self:
        unknown = cls.version.dtypes.Unknown
        dtype_pa = None if dtype == unknown else fn.dtype_native(dtype, cls.version)
        return cls.from_native(fn.lit(value, dtype_pa), name)

    @classmethod
    def from_series(cls, series: Series) -> Self:
        if len(series) == 1:
            return cls.from_native(series.native[0], series.name)
        if len(series) == 0:
            return cls.from_python(None, series.name, dtype=series.dtype)
        msg = f"Too long {len(series)!r}"
        raise InvalidOperationError(msg)

    @classmethod
    def from_unknown(
        cls, value: Any, name: str = "literal", /, *, dtype: IntoDType | None = None
    ) -> Self:
        if isinstance(value, pa.Scalar):
            return cls.from_native(value, name)
        if is_python_literal(value):
            return cls.from_python(value, name, dtype=dtype)
        native = fn.lit(value, fn.dtype_native(dtype, cls.version))
        return cls.from_native(native, name)

    def _dispatch_expr(self, node: ir.ExprIR, frame: Frame, name: str) -> Series:
        msg = f"Expected unreachable, but hit at: {node!r}"
        raise InvalidOperationError(msg)

    def _with_native(self, native: Any, name: str, /) -> Self:
        return self.from_native(native, name or self.name)

    @property
    def native(self) -> NativeScalar:
        return self._evaluated

    def to_series(self) -> Series:
        return self.broadcast(1)

    def to_python(self) -> PythonLiteral:
        result: PythonLiteral = self.native.as_py()
        return result

    def broadcast(self, length: int) -> Series:
        scalar = self.native
        if length == 1:
            chunked = fn.chunked_array(scalar)
        else:
            chunked = fn.chunked_array(fn.repeat_unchecked(scalar, length))
        return Series.from_native(chunked, self.name)

    def count(self, node: Count, frame: Frame, name: str, /) -> Scalar:
        native = node.expr.dispatch(self, frame, name).native
        return self._with_native(pa.scalar(1 if native.is_valid else 0), name)

    @unary.partial
    def null_count(self, f: F.NullCount, previous: NativeScalar) -> NativeScalar:
        return pa.scalar(0 if previous.is_valid else 1)

    def drop_nulls(  # type: ignore[override]
        self, node: FExpr[F.DropNulls], frame: Frame, name: str
    ) -> Scalar | Expr:
        previous = node.dispatch_arg(self, frame, name)
        if previous.native.is_valid:
            return previous
        chunked = fn.chunked_array([], previous.native.type)
        return ArrowExpr.from_native(chunked, name)

    @property
    def cat(self) -> ArrowCatNamespace[Scalar]:
        return ArrowCatNamespace(self)

    @property
    def list(self) -> ArrowListNamespace[Scalar]:
        return ArrowListNamespace(self)

    @property
    def str(self) -> ArrowStringNamespace[Scalar]:
        return ArrowStringNamespace(self)

    @property
    def struct(self) -> ArrowStructNamespace[Scalar]:
        return ArrowStructNamespace(self)

    dt = not_implemented()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]

    filter = not_implemented()
    over = not_implemented()
    over_ordered = not_implemented()
    map_batches = not_implemented()
    rank = not_implemented()
    # length_preserving
    rolling_sum = not_implemented()
    rolling_mean = not_implemented()
    rolling_std = not_implemented()
    rolling_var = not_implemented()
    diff = not_implemented()
    cum_sum = not_implemented()  # TODO @dangotbanned: is this just self?
    cum_count = not_implemented()
    cum_min = not_implemented()
    cum_max = not_implemented()
    cum_prod = not_implemented()


ExprOrScalarT_co = TypeVar("ExprOrScalarT_co", "ArrowExpr", "ArrowScalar", covariant=True)


class ArrowAccessor(Generic[ExprOrScalarT_co]):
    def __init__(self, compliant: ExprOrScalarT_co, /) -> None:
        self._compliant: ExprOrScalarT_co = compliant

    @property
    def compliant(self) -> ExprOrScalarT_co:
        return self._compliant

    def __narwhals_namespace__(self) -> Namespace:
        return Namespace()

    def with_native(self, native: Native, name: str, /) -> Expr | Scalar:
        return self.compliant._with_native(native, name)


class ArrowCatNamespace(
    ExprCatNamespace["Frame", "Expr"], ArrowAccessor[ExprOrScalarT_co]
):
    get_categories = unary_accessor.no_args(fn.cat.get_categories)  # pyright: ignore[reportGeneralTypeIssues]


class ArrowListNamespace(
    ExprListNamespace["Frame", "Expr | Scalar"], ArrowAccessor[ExprOrScalarT_co]
):
    @unary_accessor.partial
    def get(self, f: lists.Get, previous: Native) -> Native:
        return fn.list.get(previous, f.index)

    @unary_accessor.partial
    def join(self, f: lists.Join, previous: Native) -> Native:
        if isinstance(previous, pa.ChunkedArray):
            return fn.list.join(previous, f.separator, ignore_nulls=f.ignore_nulls)
        return fn.list.join_scalar(previous, f.separator, ignore_nulls=f.ignore_nulls)

    def contains(
        self, node: FExpr[lists.Contains], frame: Frame, name: str, /
    ) -> Expr | Scalar:
        expr, other = node.input
        prev = expr.dispatch(self.compliant, frame, name)
        item = other.dispatch(self.compliant, frame, name)
        if isinstance(item, ArrowExpr):
            # Maybe one day, not now
            raise NotImplementedError
        return self.with_native(fn.list.contains(prev.native, item.native), name)

    def aggregate(
        self, node: FExpr[lists.Aggregation], frame: Frame, name: str, /
    ) -> Expr | Scalar:
        previous = node.input[0].dispatch(self.compliant, frame, name)
        agg = group_by.AggSpec._from_list_agg(node.function, "values")
        return self.with_native(agg.agg_list(previous.native), name)

    @unary_accessor.partial
    def sort(self, f: lists.Sort, previous: Native) -> Native:
        opt = f.options
        if isinstance(previous, pa.Scalar):
            return fn.list.sort_scalar(previous, opt)
        return fn.list.sort(
            previous, descending=opt.descending, nulls_last=opt.nulls_last
        )

    min = aggregate
    max = aggregate
    mean = aggregate
    median = aggregate
    sum = aggregate
    any = aggregate
    all = aggregate
    first = aggregate
    last = aggregate
    n_unique = aggregate
    len = unary_accessor.no_args(fn.list.len)  # pyright: ignore[reportGeneralTypeIssues]
    unique = unary_accessor.no_args(fn.list.unique)  # pyright: ignore[reportGeneralTypeIssues]


class ArrowStringNamespace(
    ExprStringNamespace["Frame", "Expr | Scalar"], ArrowAccessor[ExprOrScalarT_co]
):
    @unary_accessor.partial
    def slice(self, f: strings.Slice, previous: Native) -> Native:
        return fn.str.slice(previous, f.offset, f.length)

    @unary_accessor.partial
    def zfill(self, f: strings.ZFill, previous: Native) -> Native:
        return fn.str.zfill(previous, f.length)

    @unary_accessor.partial
    def contains(self, f: strings.Contains, previous: Native) -> Native:
        return fn.str.contains(previous, f.pattern, literal=f.literal)

    @unary_accessor.partial
    def ends_with(self, f: strings.EndsWith, previous: Native) -> Native:
        return fn.str.ends_with(previous, f.suffix)

    def replace(
        self, node: FExpr[strings.Replace], frame: Frame, name: str, /
    ) -> Expr | Scalar:
        func = node.function
        pattern, literal, n = (func.pattern, func.literal, func.n)
        expr, other = node.input
        prev = expr.dispatch(self.compliant, frame, name)
        value = other.dispatch(self.compliant, frame, name)
        if isinstance(value, ArrowScalar):
            result = fn.str.replace(
                prev.native, pattern, value.native.as_py(), literal=literal, n=n
            )
        elif isinstance(prev, ArrowExpr):
            result = fn.str.replace_vector(
                prev.native, pattern, value.native, literal=literal, n=n
            )
        else:
            # not sure this even makes sense
            msg = "TODO: `ArrowScalar.str.replace(value: ArrowExpr)`"
            raise NotImplementedError(msg)
        return self.with_native(result, name)

    def replace_all(
        self, node: FExpr[strings.ReplaceAll], frame: Frame, name: str, /
    ) -> Expr | Scalar:
        rewrite: FExpr[Any] = common.replace(
            node, function=node.function.to_replace_n(-1)
        )
        return self.replace(rewrite, frame, name)

    @unary_accessor.partial
    def split(self, f: strings.Split, previous: Native) -> Native:
        return fn.str.split(previous, f.by)

    @unary_accessor.partial
    def starts_with(self, f: strings.StartsWith, previous: Native) -> Native:
        return fn.str.starts_with(previous, f.prefix)

    @unary_accessor.partial
    def strip_chars(self, f: strings.StripChars, previous: Native) -> Native:
        return fn.str.strip_chars(previous, f.characters)

    len_chars = unary_accessor.no_args(fn.str.len_chars)  # pyright: ignore[reportGeneralTypeIssues]
    to_uppercase = unary_accessor.no_args(fn.str.to_uppercase)  # pyright: ignore[reportGeneralTypeIssues]
    to_lowercase = unary_accessor.no_args(fn.str.to_lowercase)  # pyright: ignore[reportGeneralTypeIssues]
    to_titlecase = unary_accessor.no_args(fn.str.to_titlecase)  # pyright: ignore[reportGeneralTypeIssues]
    to_date = not_implemented()
    to_datetime = not_implemented()


class ArrowStructNamespace(
    ExprStructNamespace["Frame", "Expr | Scalar"], ArrowAccessor[ExprOrScalarT_co]
):
    @unary_accessor.partial
    def field(self, f: FieldByName, previous: Native) -> Native:
        return fn.struct.field(previous, f.name)
