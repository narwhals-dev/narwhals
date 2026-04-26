"""Useful types for `narwhals._plan.compliant`.

## Notes
- This module has 0 runtime dependencies on the rest of `compliant.*`
- `Native*` type vars defined here *do not* have defaults
    - If you need those, use `narwhals._plan.typing` instead
"""

# ruff: noqa: PLC0105
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan import expressions as ir
from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import TypeAlias

    from narwhals._native import NativeDataFrame, NativeSeries
    from narwhals._plan.compliant.dataframe import (
        CompliantDataFrame,
        CompliantFrame,
        EagerDataFrame,
    )
    from narwhals._plan.compliant.expr import CompliantExpr, EagerExpr
    from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
    from narwhals._plan.compliant.namespace import CompliantNamespace
    from narwhals._plan.compliant.scalar import CompliantScalar, EagerScalar
    from narwhals._plan.compliant.series import CompliantSeries


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

ExprAny: TypeAlias = "CompliantExpr[Any, Any, Any]"
ScalarAny: TypeAlias = "CompliantScalar[Any, Any, Any]"
SeriesAny: TypeAlias = "CompliantSeries[Any]"
FrameAny: TypeAlias = "CompliantFrame[Any]"
DataFrameAny: TypeAlias = "CompliantDataFrame[Any, Any]"
LazyFrameAny: TypeAlias = "CompliantLazyFrame[Any]"
NamespaceAny: TypeAlias = "CompliantNamespace[Any, Any, Any]"
Namespace: TypeAlias = "CompliantNamespace[FrameT, ExprT_co, ScalarT_co]"

EagerExprAny: TypeAlias = "EagerExpr[Any, Any, Any, Any]"
EagerScalarAny: TypeAlias = "EagerScalar[Any, Any, Any, Any]"
EagerDataFrameAny: TypeAlias = "EagerDataFrame[Any, Any]"

ExprT_co = TypeVar("ExprT_co", bound=ExprAny, covariant=True)
ScalarT_co = TypeVar(
    "ScalarT_co", bound="ExprAny | ScalarAny", covariant=True, default=ExprT_co
)
SeriesT = TypeVar("SeriesT", bound=SeriesAny)
SeriesT_co = TypeVar("SeriesT_co", bound=SeriesAny, covariant=True)
FrameT = TypeVar("FrameT", bound=FrameAny)
FrameT_co = TypeVar("FrameT_co", bound=FrameAny, covariant=True)
FrameT_contra = TypeVar("FrameT_contra", bound=FrameAny, contravariant=True)
DataFrameT = TypeVar("DataFrameT", bound=DataFrameAny)
DataFrameT_co = TypeVar("DataFrameT_co", bound=DataFrameAny, covariant=True)
LazyFrameT = TypeVar("LazyFrameT", bound=LazyFrameAny)
LazyFrameT_co = TypeVar("LazyFrameT_co", bound=LazyFrameAny, covariant=True)
NamespaceT_co = TypeVar("NamespaceT_co", bound="NamespaceAny", covariant=True)

EagerExprT_co = TypeVar("EagerExprT_co", bound=EagerExprAny, covariant=True)
EagerScalarT_co = TypeVar(
    "EagerScalarT_co",
    bound="EagerExprAny | EagerScalarAny",
    covariant=True,
    default=EagerExprT_co,
)
EagerDataFrameT = TypeVar("EagerDataFrameT", bound=EagerDataFrameAny)
EagerDataFrameT_co = TypeVar(
    "EagerDataFrameT_co", bound=EagerDataFrameAny, covariant=True
)


class SupportsNarwhalsNamespace(Protocol[NamespaceT_co]):
    def __narwhals_namespace__(self) -> NamespaceT_co: ...


class CanNamespace(Protocol[FrameT, ExprT_co, ScalarT_co]):
    """Use this instead of `Namespace` or `*Frame`, when all that's needed is access to a namespace."""

    def __narwhals_namespace__(self) -> Namespace[FrameT, ExprT_co, ScalarT_co]: ...


# - `Self_` and `Frame` need to share a `*Namespace`
# - `Self_` needs to express how we get to `CompliantExpr` with it's bound
Self_ = TypeVar("Self_", contravariant=True)
Frame = TypeVar("Frame", bound=FrameAny, contravariant=True)
IR = TypeVar("IR", bound="ir.ExprIR", contravariant=True)
F_contra = TypeVar("F_contra", bound="ir.Function", contravariant=True)

R = TypeVar("R", bound=ExprAny, covariant=True)


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


# NOTE: Very important that these stay covariant!
ET_co = TypeVar("ET_co", bound="ExprAny", covariant=True)
"""`CompliantExpr`"""
ST_co = TypeVar("ST_co", bound="ExprAny | ScalarAny", covariant=True)
"""`CompliantScalar`"""


def dispatch_function_expr_from_namespace(
    self: Namespace[Frame, ET_co, ST_co], node: ir.FunctionExpr, frame: Frame, name: str
) -> Iterator[ET_co | ST_co]:
    """`ArrowNamespace` horizontal and range functions use `_expr.from_ir` on their inputs.

    basically what I had in `_parameters.Variadic.iter_dispatch_args`
    """
    ns = self.__narwhals_namespace__()
    expr_irs = iter(node.input)
    yield ns.from_ir(next(expr_irs), frame, name)
    for expr_ir in expr_irs:
        yield ns.from_ir(expr_ir, frame, "")


def dispatch_from_scalar_ensure_scalar(
    ctx: DispatchScopeScalar[Frame, ST_co], node: ir.ExprIR, frame: Frame, name: str
) -> ST_co:
    result = node.dispatch(ctx, frame, name)
    ns = ctx.__narwhals_namespace__()
    scalar = ns._scalar
    if isinstance(result, scalar):
        return result
    raise NotImplementedError


# TODO @dangotbanned: Decide on a better name
class DispatchScope(Protocol[NamespaceT_co, ET_co]):
    """Represents either `*Expr` or `*Namespace`.

    E.g. the widest possible type you can dispatch from.
    """

    def __narwhals_expr_prepare__(self) -> ET_co:
        """Return a partially initialized `CompliantExpr`.

        ## Notes
        - The only external (narwhals-level) requirement is that we have an instance to call methods on
        - If there are any other bits of state you need in an implementation, add them here
        - Defaults
            - Namespace -> Expr
            - Expr -> Expr
            - Scalar -> Scalar
        - Needed because dispatching starts at the last expression
            - Whether that's a good choice is up in the air
        """
        ...

    def __narwhals_namespace__(self) -> NamespaceT_co: ...


DispatchScopeAny: TypeAlias = (
    "DispatchScope[Namespace[Frame, ET_co, ST_co], ET_co | ST_co]"
)
DispatchScopeExpr: TypeAlias = "DispatchScope[Namespace[Frame, ET_co, Any], ET_co]"
DispatchScopeScalar: TypeAlias = "DispatchScope[Namespace[Frame, Any, ST_co], ST_co]"


class CallNamespace(Protocol):
    def __call__(self, obj: DispatchScope[NamespaceT_co, ET_co], /) -> NamespaceT_co: ...


class CallExprPrepare(Protocol):
    def __call__(self, obj: DispatchScope[NamespaceT_co, ET_co], /) -> ET_co: ...


class GetMethod(Protocol):
    def __call__(
        self, obj: ET_co | ST_co | Namespace[Frame, ET_co, ST_co], /
    ) -> BoundMethod[Any, Frame, ET_co | ST_co]: ...


ExprIR_contra = TypeVar("ExprIR_contra", bound="ir.ExprIR", contravariant=True)


class Binder(Protocol[ExprIR_contra]):
    """The type of `ExprIR.__expr_ir_dispatch__.bind`.

    - Takes a context that we try to access the method on.
    - This step can fail (`AttributeError`), and that represents a different kind of error than if the call failed
    """

    def __call__(
        self, ctx: DispatchScopeAny[Frame, ET_co, ST_co], /
    ) -> BoundMethod[ExprIR_contra, Frame, ET_co | ST_co]: ...


class BoundMethod(Protocol[ExprIR_contra, Frame, ET_co]):
    """The return type of `ExprIR.__expr_ir_dispatch__.bind`.

    - `None` can be returned when subclassing `*Expr`, but not implementing the method
    """

    def __call__(
        self, node: ExprIR_contra, frame: Frame, name: str, /
    ) -> ET_co | None: ...
