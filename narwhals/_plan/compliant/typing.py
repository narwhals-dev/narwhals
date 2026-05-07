"""Useful types for `narwhals._plan.compliant`.

## Notes
- This module has 0 runtime dependencies on the rest of `compliant.*`
- `Native*` type vars defined here *do not* have defaults
    - If you need those, use `narwhals._plan.typing` instead
"""

# ruff: noqa: PLC0105
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, overload

from narwhals._plan import expressions as ir
from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from typing_extensions import TypeAlias

    from narwhals._native import NativeDataFrame, NativeSeries
    from narwhals._plan.compliant import classes as cc
    from narwhals._plan.compliant.dataframe import (
        CompliantDataFrame,
        CompliantFrame,
        EagerDataFrame,
    )
    from narwhals._plan.compliant.expr import CompliantColumn, CompliantExpr, EagerExpr
    from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
    from narwhals._plan.compliant.namespace import CompliantNamespace
    from narwhals._plan.compliant.scalar import CompliantScalar, EagerScalar
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.plans.visitors import ResolvedToCompliant


Native = TypeVar("Native")
"""Unbounded type variable, representing *any* native object.

Assume nothing, permit anything; rely on well-defined protocols to do the talking.
"""
FromNative = TypeVar("FromNative")
"""Same as `Native`, but should be scoped to constructor method(s) and not the class."""

Native_co = TypeVar("Native_co", covariant=True)

NativeExpr_co = TypeVar("NativeExpr_co", covariant=True)
"""The type of `CompliantExpr.native`.

This can be literally anything, but some typical candidates would be:
- A native expression representation
- A native series or array
- A native scalar
"""

NativeScalar_co = TypeVar("NativeScalar_co", covariant=True)
NativeSeriesT = TypeVar("NativeSeriesT", bound="NativeSeries")
"""Be careful using this type var!

- For broadcasting, it seems to be unavoidable to have *some* invariance
- Try to keep `NativeSeriesT_co` in as many places as possible
"""

NativeSeriesT_co = TypeVar("NativeSeriesT_co", bound="NativeSeries", covariant=True)
NativeDataFrameT = TypeVar("NativeDataFrameT", bound="NativeDataFrame")
NativeDataFrameT_co = TypeVar(
    "NativeDataFrameT_co", bound="NativeDataFrame", covariant=True
)

PlanEvaluator: TypeAlias = "ResolvedToCompliant[Native]"

ColumnAny: TypeAlias = "CompliantColumn[Any, Any]"
ExprAny: TypeAlias = "CompliantExpr[Any, Any, Any]"
ScalarAny: TypeAlias = "CompliantScalar[Any, Any, Any]"
Series: TypeAlias = "CompliantSeries[NativeSeriesT_co]"
SeriesAny: TypeAlias = "CompliantSeries[Any]"
FrameAny: TypeAlias = "CompliantFrame[Any]"
DataFrame: TypeAlias = "CompliantDataFrame[NativeDataFrameT_co, NativeSeriesT_co]"
DataFrameAny: TypeAlias = "CompliantDataFrame[Any, Any]"
LazyFrame: TypeAlias = "CompliantLazyFrame[Native_co]"
LazyFrameAny: TypeAlias = "CompliantLazyFrame[Any]"
NamespaceAny: TypeAlias = "CompliantNamespace[Any, Any, Any]"
Namespace: TypeAlias = "CompliantNamespace[FrameT, ExprT_co, ScalarT_co]"
PlanEvaluatorAny: TypeAlias = "PlanEvaluator[Any]"

EagerExprAny: TypeAlias = "EagerExpr[Any, Any, Any, Any]"
EagerScalarAny: TypeAlias = "EagerScalar[Any, Any, Any]"
EagerDataFrameAny: TypeAlias = "EagerDataFrame[Any, Any]"

ExprT_co = TypeVar("ExprT_co", bound=ExprAny, covariant=True)
"""Covariant TypeVar for `CompliantExpr`."""
ScalarT_co = TypeVar(
    "ScalarT_co", bound="ExprAny | ScalarAny", covariant=True, default=ExprT_co
)
"""Covariant TypeVar for `CompliantScalar`.

Defaults to the one used for `CompliantExpr`.
"""
ScalarNoDefaultT_co = TypeVar(
    "ScalarNoDefaultT_co", bound="ExprAny | ScalarAny", covariant=True
)
"""Covariant TypeVar for `CompliantScalar`.

Does not use a default due to [`mypy` PEP 696 bugs].

[`mypy` PEP 696 bugs]: https://github.com/python/mypy/issues?q=sort%3Aupdated-desc%20is%3Aissue%20is%3Aopen%20label%3Atopic-pep-696
"""
SeriesT = TypeVar("SeriesT", bound=SeriesAny)
SeriesT_co = TypeVar("SeriesT_co", bound=SeriesAny, covariant=True)
"""Covariant TypeVar for `CompliantSeries`."""
FrameT = TypeVar("FrameT", bound=FrameAny)
FrameT_co = TypeVar("FrameT_co", bound=FrameAny, covariant=True)
FrameT_contra = TypeVar("FrameT_contra", bound=FrameAny, contravariant=True)
DataFrameT = TypeVar("DataFrameT", bound=DataFrameAny)
DataFrameT_co = TypeVar("DataFrameT_co", bound=DataFrameAny, covariant=True)
"""Covariant TypeVar for `CompliantDataFrame`."""
LazyFrameT = TypeVar("LazyFrameT", bound=LazyFrameAny)
LazyFrameT_co = TypeVar("LazyFrameT_co", bound=LazyFrameAny, covariant=True)
"""Covariant TypeVar for `CompliantLazyFrame`."""
NamespaceT_co = TypeVar("NamespaceT_co", bound="NamespaceAny", covariant=True)

EagerExprT_co = TypeVar("EagerExprT_co", bound=EagerExprAny, covariant=True)
EagerScalarT_co = TypeVar(
    "EagerScalarT_co", bound="EagerExprAny | EagerScalarAny", covariant=True
)
EagerDataFrameT = TypeVar("EagerDataFrameT", bound=EagerDataFrameAny)
EagerDataFrameT_co = TypeVar(
    "EagerDataFrameT_co", bound=EagerDataFrameAny, covariant=True
)
EagerDataFrameT_contra = TypeVar(
    "EagerDataFrameT_contra", bound="EagerDataFrameAny", contravariant=True
)

PlanEvaluatorT_co = TypeVar("PlanEvaluatorT_co", bound="PlanEvaluatorAny", covariant=True)
"""Covariant TypeVar for `ResolvedToCompliant`.

Provides the conversion:

    ResolvedPlan -> CompliantLazyFrame[Native]
"""


class CanNamespace(Protocol[FrameT, ExprT_co, ScalarT_co]):
    """Use this instead of `Namespace` or `*Frame`, when all that's needed is access to a namespace."""

    def __narwhals_namespace__(self) -> Namespace[FrameT, ExprT_co, ScalarT_co]: ...


# NOTE: Very important that these stay covariant!
ColumnT_co = TypeVar("ColumnT_co", bound="ColumnAny", covariant=True)
"""Any column."""

LF = TypeVar("LF", bound=LazyFrameAny, covariant=True)
PE = TypeVar("PE", bound="PlanEvaluatorAny", covariant=True)
DF = TypeVar("DF", bound=DataFrameAny, covariant=True)
S = TypeVar("S", bound=SeriesAny, covariant=True)
E = TypeVar("E", bound="ExprAny", covariant=True)
"""A column representing `.expr`."""
SC = TypeVar("SC", bound="ExprAny | ScalarAny", covariant=True)
"""A column representing `.scalar`."""

ET_co = TypeVar("ET_co", bound="ExprAny", covariant=True)
"""`CompliantExpr`"""
ST_co = TypeVar("ST_co", bound="ExprAny | ScalarAny", covariant=True)
"""`CompliantScalar`"""
# - `Self_` and `Frame` need to share a `*Namespace`
# - `Self_` needs to express how we get to `CompliantExpr` with it's bound
Self_ = TypeVar("Self_", contravariant=True)
Frame = TypeVar("Frame", bound=FrameAny, contravariant=True)
Frame2 = TypeVar("Frame2", bound="DataFrameAny | LazyFrameAny", contravariant=True)
"""`CompliantDataFrame | CompliantLazyFrame`.

`Frame` was taken already :sad:
"""

IR = TypeVar("IR", bound="ir.ExprIR", contravariant=True)
F_contra = TypeVar("F_contra", bound="ir.Function", contravariant=True)

R = TypeVar("R", bound=ColumnAny, covariant=True)


class ExprMethod(Protocol[Self_, IR, Frame, R]):
    """A (unbound) `CompliantExpr` or namespace accessor method.

    That is, this describes the method *without* an instance.
    """

    def __call__(_self, self: Self_, node: IR, frame: Frame, name: str, /) -> R:
        """Bind and evaluate an expression.

        Arguments:
            self: An object providing a route to a `CompliantExpr` instance.
            node: The expression to evaluate.
            frame: The `CompliantFrame` context for the expression.
            name: The output column name (see `NamedIR.name`).

        Note:
            Ignore `_self`, see https://github.com/python/mypy/issues/16200
        """
        ...


class BoundExprMethod(Protocol[IR, Frame, R]):
    """A `CompliantExpr` or namespace accessor method, after binding `self`.

    ## Notes
    - Current version binds to the namespace accessor
        - That makes sense in the context of this being a "method"
    - The accessor is only being kept around for typing
        - which isn't working great anyway
    - The wrapper functions would be identical if `unary_accessor.__get__` did
        - `return MethodType(self._wrapper_method, instance.compliant)`
    """

    def __call__(self, node: IR, frame: Frame, name: str, /) -> R: ...


# NOTE: Equivalent to writing a sub-protocol
# runtime requires that `FunctionExpr` is not a ForwardRef
FunctionImplMethod = ExprMethod[Self_, ir.FunctionExpr[F_contra], Frame, R]
BoundFunctionImplMethod = BoundExprMethod[ir.FunctionExpr[F_contra], Frame, R]


# TODO @dangotbanned: Decide on a better name
class DispatchScope(Protocol[NamespaceT_co, ColumnT_co]):
    """Represents either `*Expr` or `*Namespace`.

    E.g. the widest possible type you can dispatch from.
    """

    def __narwhals_expr_prepare__(self) -> ColumnT_co:
        """Return a partially initialized `CompliantExpr`.

        ## Notes
        - The only external (narwhals-level) requirement is that we have an instance to call methods on
        - Defaults
            - Namespace -> Expr
            - Expr -> Expr
            - Scalar -> Scalar
        - Needed because dispatching starts at the last expression
            - Whether that's a good choice is up in the air
        """
        ...

    def __narwhals_namespace__(self) -> NamespaceT_co: ...


C = TypeVar("C", bound="cc.ClassesAny", covariant=True)


class DispatchScope2(Protocol[C, ColumnT_co]):
    @property
    def __narwhals_classes__(self) -> C: ...
    def __narwhals_expr_prepare__(self) -> ColumnT_co: ...


class HasExpr(Protocol[E]):
    @property
    def _expr(self) -> type[E]: ...


class HasScalar(Protocol[SC]):
    @property
    def _scalar(self) -> type[SC]: ...


DispatchScopeAny: TypeAlias = (
    "DispatchScope[Namespace[Frame, ET_co, ST_co], ET_co | ST_co]"
)

FrameUnknown: TypeAlias = (
    "cc.EagerClasses[Any, Any, E, SC] | cc.LazyClasses[Any, Any, E, SC]"
)

DispatchEager: TypeAlias = "DispatchScope2[cc.EagerClasses[DF, Any, E, SC], E | SC]"
DispatchLazy: TypeAlias = "DispatchScope2[cc.LazyClasses[LF, Any, E, SC], E | SC]"

DispatchUnknown: TypeAlias = "DispatchScope2[FrameUnknown[E, SC], E | SC]"
"""For whatever reason, an overload couldn't match `frame`.

This *should* preserve the typing for `*Expr` & `*Scalar` as a fallback.
"""

DispatchAny: TypeAlias = (
    "DispatchEager[DF, E, SC] | DispatchLazy[LF, E, SC] | DispatchUnknown[E, SC]"
)

BoundMethod2Any: TypeAlias = "BoundMethod2[IR, DF, E | SC] | BoundMethod2[IR, LF, E | SC] | BoundMethod2[IR, Any, E | SC]"


def binder_actual(f1: CallExprPrepare2, f2: GetMethod2, /) -> BinderBind2:
    """`_dispatch._binder`.

    - Uses `__narwhals_classes__`
    - Aiming to support eager and lazy via the same API
    - From the dispatcher side, it's just about passing the frame to the new caller
        - it doesn't matter really what we were given
    """

    @overload
    def bind(ctx: DispatchEager[DF, E, SC], /) -> BoundMethod2[Any, DF, E | SC]: ...
    @overload
    def bind(ctx: DispatchLazy[LF, E, SC], /) -> BoundMethod2[Any, LF, E | SC]: ...
    @overload
    def bind(ctx: DispatchUnknown[E, SC], /) -> BoundMethod2[Any, Any, E | SC]: ...
    def bind(ctx: DispatchScope2[Incomplete, Incomplete], /) -> Incomplete:
        return f2(f1(ctx))

    return bind


class BinderBind2(Protocol):
    """Overloaded now to accurately represent eager/lazy support."""

    @overload
    def __call__(
        self, ctx: DispatchEager[DF, E, SC], /
    ) -> BoundMethod2[Any, DF, E | SC]: ...
    @overload
    def __call__(
        self, ctx: DispatchLazy[LF, E, SC], /
    ) -> BoundMethod2[Any, LF, E | SC]: ...
    @overload
    def __call__(
        self, ctx: DispatchUnknown[E, SC], /
    ) -> BoundMethod2[Any, Any, E | SC]: ...
    @overload
    def __call__(
        self, ctx: DispatchAny[DF, E, SC, LF], /
    ) -> BoundMethod2Any[Any, DF, E, SC, LF]: ...
    def __call__(
        self, ctx: DispatchAny[Any, Any, Any, Any], /
    ) -> BoundMethod2[Any, Any, Any]: ...


class Binder2(Protocol[IR]):
    """Gets the `IR` once back inside `Dispatcher`."""

    @overload
    def __call__(
        self, ctx: DispatchEager[DF, E, SC], /
    ) -> BoundMethod2[IR, DF, E | SC]: ...
    @overload
    def __call__(
        self, ctx: DispatchLazy[LF, E, SC], /
    ) -> BoundMethod2[IR, LF, E | SC]: ...
    @overload
    def __call__(
        self, ctx: DispatchUnknown[E, SC], /
    ) -> BoundMethod2[IR, Any, E | SC]: ...
    @overload
    def __call__(
        self, ctx: DispatchAny[DF, E, SC, LF], /
    ) -> BoundMethod2Any[IR, DF, E, SC, LF]: ...
    def __call__(
        self, ctx: DispatchAny[Any, Any, Any, Any], /
    ) -> BoundMethod2[IR, Any, Any]: ...


class CallNamespace(Protocol):
    def __call__(
        self, obj: DispatchScope[NamespaceT_co, ColumnT_co], /
    ) -> NamespaceT_co: ...


class CallExprPrepare(Protocol):
    def __call__(
        self, obj: DispatchScope[NamespaceT_co, ColumnT_co], /
    ) -> ColumnT_co: ...


class CallExprPrepare2(Protocol):
    """`_binder.f1`."""

    def __call__(self, ctx: DispatchScope2[cc.C, ColumnT_co], /) -> ColumnT_co: ...


class GetMethod(Protocol):
    #  `mypy`: Cannot use a covariant type variable as a parameter
    def __call__(self, obj: ColumnT_co, /) -> BoundMethod[Any, Any, ColumnT_co]: ...  # type: ignore[misc]


class GetMethod2(Protocol):
    def __call__(self, obj: ColumnT_co, /) -> BoundMethod2[Any, Any, ColumnT_co]: ...  # type: ignore[misc]


class GetClassMethod(Protocol):
    def __call__(self, tp: type[E | SC], /) -> BoundMethod[Any, Any, E | SC]: ...


class GetClassMethod2(Protocol):
    def __call__(self, tp: type[E | SC], /) -> BoundMethod2[Any, Any, E | SC]: ...


class GetExpr(Protocol):
    def __call__(self, obj: HasExpr[ET_co], /) -> type[ET_co]: ...


class GetScalar(Protocol):
    def __call__(self, obj: HasScalar[ST_co], /) -> type[ST_co]: ...


class Binder(Protocol[IR]):
    """The type of `ExprIR.__expr_ir_dispatch__.bind`.

    - Takes a context that we try to access the method on.
    - This step can fail (`AttributeError`), and that represents a different kind of error than if the call failed
    """

    def __call__(
        self, ctx: DispatchScopeAny[Frame, ET_co, ST_co], /
    ) -> BoundMethod[IR, Frame, ET_co | ST_co]: ...


class BoundMethod(Protocol[IR, Frame, ColumnT_co]):
    """The return type of `ExprIR.__expr_ir_dispatch__.bind`.

    - `None` can be returned when subclassing `*Expr`, but not implementing the method
    """

    def __call__(self, node: IR, frame: Frame, name: str, /) -> ColumnT_co | None: ...


class BoundMethod2(Protocol[IR, Frame2, ColumnT_co]):
    def __call__(self, node: IR, frame: Frame2, name: str, /) -> ColumnT_co | None: ...
