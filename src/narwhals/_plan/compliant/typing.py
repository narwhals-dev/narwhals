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
    from typing import TypeAlias

    from narwhals._native import NativeDataFrame, NativeSeries
    from narwhals._plan.compliant import classes as cc
    from narwhals._plan.compliant.dataframe import CompliantDataFrame, CompliantFrame
    from narwhals._plan.compliant.expr import CompliantColumn, CompliantExpr
    from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
    from narwhals._plan.compliant.scalar import CompliantScalar
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.plans.visitors import ResolvedToCompliant


Native = TypeVar("Native")
"""Unbounded type variable, representing *any* native object.

Assume nothing, permit anything; rely on well-defined protocols to do the talking.
"""
FromNative = TypeVar("FromNative")
"""Same as `Native`, but should be scoped to constructor method(s) and not the class."""

Native_co = TypeVar("Native_co", covariant=True)
NativeColumn_co = TypeVar("NativeColumn_co", covariant=True)
"""The type of `CompliantColumn.native`.

Where `*Expr` and `*Scalar` fill this slot with *their* respective type parameter.

This avoids the (expected) issue of using incompatible types for each, while allowing
all 3 native types to be the same if desired.
"""


NativeExpr_co = TypeVar("NativeExpr_co", covariant=True)
"""The type of `CompliantExpr.native`.

This can be literally anything, but some typical candidates would be:
- A native expression representation
- A native series or array
- A native scalar
"""

NativeScalar_co = TypeVar("NativeScalar_co", covariant=True)
"""The type of `CompliantScalar.native`."""

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


Column: TypeAlias = "CompliantColumn[DeprecatedFrameT_contra, NativeColumn_co, NativeExpr_co, NativeScalar_co]"
Series: TypeAlias = "CompliantSeries[NativeSeriesT_co]"
DataFrame: TypeAlias = "CompliantDataFrame[NativeDataFrameT_co, NativeSeriesT_co]"
LazyFrame: TypeAlias = "CompliantLazyFrame[Native_co]"
Frame: TypeAlias = (
    "DataFrame[NativeDataFrameT_co, NativeSeriesT_co] | LazyFrame[Native_co]"
)
PlanEvaluator: TypeAlias = "ResolvedToCompliant[Native]"

ColumnAny: TypeAlias = "CompliantColumn[Any, Any, Any, Any]"
ExprAny: TypeAlias = "CompliantExpr[Any, Any, Any]"
ScalarAny: TypeAlias = "CompliantScalar[Any, Any, Any]"
SeriesAny: TypeAlias = "CompliantSeries[Any]"
DataFrameAny: TypeAlias = "CompliantDataFrame[Any, Any]"
LazyFrameAny: TypeAlias = "CompliantLazyFrame[Any]"
FrameAny: TypeAlias = "DataFrameAny | LazyFrameAny"
PlanEvaluatorAny: TypeAlias = "PlanEvaluator[Any]"

SeriesT = TypeVar("SeriesT", bound=SeriesAny)
DataFrameT = TypeVar("DataFrameT", bound=DataFrameAny)
FrameT = TypeVar("FrameT", bound="FrameAny")
FrameT_contra = TypeVar("FrameT_contra", bound="FrameAny", contravariant=True)
"""Contravariant TypeVar for `CompliantDataFrame | CompliantLazyFrame`."""

# NOTE: Very important that these stay covariant!
FrameT_co = TypeVar("FrameT_co", bound="FrameAny", covariant=True)
"""Covariant TypeVar for `CompliantDataFrame | CompliantLazyFrame`."""
ColumnT_co = TypeVar("ColumnT_co", bound="ColumnAny", covariant=True)
"""Any column."""
LF = TypeVar("LF", bound=LazyFrameAny, covariant=True)
"""Covariant TypeVar for `CompliantLazyFrame`."""
PE = TypeVar("PE", bound="PlanEvaluatorAny", covariant=True)
"""Covariant TypeVar for `ResolvedToCompliant`.

Provides the conversion:

    ResolvedPlan -> CompliantLazyFrame[Native]
"""
DF = TypeVar("DF", bound=DataFrameAny, covariant=True)
"""Covariant TypeVar for `CompliantDataFrame`."""
S = TypeVar("S", bound=SeriesAny, covariant=True)
"""Covariant TypeVar for `CompliantSeries`."""
E = TypeVar("E", bound="ExprAny", covariant=True)
"""Covariant TypeVar for `CompliantExpr`."""
SC = TypeVar("SC", bound="ExprAny | ScalarAny", covariant=True)
"""Covariant TypeVar for `CompliantScalar`."""

Classes = TypeVar("Classes", bound="cc.CompliantClasses[Any, Any]", covariant=True)


class _Dispatchee(Protocol[Classes, ColumnT_co]):
    __slots__ = ()

    @property
    def __narwhals_classes__(self) -> Classes: ...
    # https://github.com/python/mypy/issues/15182
    def __new__(cls) -> ColumnT_co: ...  # type: ignore[misc]


class Caller(_Dispatchee["cc.CompliantClasses[E, SC]", E | SC], Protocol[E, SC]):
    """The type of the compliant context for `ExprIR` dispatch.

    `Caller` describes `self` being rescoped as `ctx` here:

        class CompliantColumn(Protocol):
            def dispatch(self, node: Any, frame: Any, name: str, /) -> Any:
                ctx: Caller[E, SC] = self
                return node.__expr_ir_dispatch__(node, ctx, frame, name)

    ## Why not `Compliant*`?
    Great question! First though, some background ...

    ### Background
    - Dispatching an `ExprIR` starts from the most recent expression (not the first)
    - Methods on `*Expr` and `*Scalar` can return the same type or transition to a different shape
        - Using `Self` is not an option
    - Implementations of `CompliantColumn` are versioned classes
        - The protocols cannot be defined with an assumption of invariance

    ### Ummmmm?
    Yeah, it's a minefield. Let's look at an example to ground ourselves:

    >>> import narwhals._plan as nw
    >>> df = nw.from_dict({"a": [3, 1, 2], "b": [1.2, 2.0, 2.3]}, backend="pyarrow")
    >>> expr = nw.col("a").sort().last()
    >>> df.select(expr)
    ┌─────────────┐
    |nw.DataFrame |
    |-------------|
    |pyarrow.Table|
    |a: int64     |
    |----         |
    |a: [[3]]     |
    └─────────────┘

    Dispatching an `ExprIR` starts from the most recent expression (not the first):
    >>> for e in expr._ir.iter_right():
    ...     print(f"{e.__expr_ir_dispatch__.name!s:<4} | {e!r}")
    last | col('a').sort(asc).last()
    sort | col('a').sort(asc)
    col  | col('a')

    Methods on `*Expr` and `*Scalar` can return the same type or transition to a different shape:
    >>> from narwhals._plan import expressions as ir
    >>> from narwhals._plan.compliant import typing as ct

    >>> # A speedrun of what happens between `DataFrame.select` and `CompliantDataFrame.select`
    >>> expr_ir = expr._ir
    >>> # ...
    >>> named_ir = ir.NamedIR("a", expr_ir)

    >>> # Now, we need to set the scene for `CompliantColumn.dispatch`
    >>> frame: ct.DataFrameAny = df._compliant
    >>> tp_expr: type[ct.ExprAny] = frame.__narwhals_classes__.expr
    >>> ctx = tp_expr.__new__(tp_expr)

    >>> print("Initial context:", ctx.__class__.__name__)
    Initial context: ArrowExpr
    >>> result = ctx.dispatch(named_ir.expr, frame, named_ir.name)
    >>> print("Final context  :", result.__class__.__name__)
    Final context  : ArrowScalar

    Implementations of `CompliantColumn` are versioned classes:
    >>> print(f"{result.__class__.__name__} ({result.version})")
    ArrowScalar (Version.MAIN)
    >>> v1 = result.__narwhals_classes__.v1.scalar
    >>> print(f"{v1.__name__} ({v1.version})")
    ArrowScalarV1 (Version.V1)

    ### So what?
    `Caller` encodes a *degree of uncertainty*, which can be reeled in by each
    implementation of `CompliantColumn.dispatch`.

    The parts defined *here* (`__narwhals_classes__`, `__new__`) are generic but use **gradual**
    typing in `CompliantColumn`.

    This concession avoids [introducing cycles], which is inherent to:
    - The idea of `__narwhals_classes__` being accessible by all of the classes it is generic over 🤯
    - Protocols representing graphs

    [introducing cycles]: https://github.com/narwhals-dev/narwhals/issues/3643
    """

    __slots__ = ()


# TODO @dangotbanned: Finish migrating to post `CompliantFrame`/`CompliantNamespace`-typing
DeprecatedFrameT_contra = TypeVar(
    "DeprecatedFrameT_contra", bound="CompliantFrame[Any]", contravariant=True
)
"""**URGENT**: Needs changing to DataFrame and LazyFrame support."""
Self_ = TypeVar("Self_", contravariant=True)
IR = TypeVar("IR", bound="ir.ExprIR", contravariant=True)
F_contra = TypeVar("F_contra", bound="ir.Function", contravariant=True)


class ExprMethod(Protocol[Self_, IR, DeprecatedFrameT_contra, ColumnT_co]):
    """A (unbound) `CompliantExpr` or namespace accessor method.

    That is, this describes the method *without* an instance.
    """

    def __call__(
        _self, self: Self_, node: IR, frame: DeprecatedFrameT_contra, name: str, /
    ) -> ColumnT_co:
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


class BoundExprMethod(Protocol[IR, DeprecatedFrameT_contra, ColumnT_co]):
    """A `CompliantExpr` or namespace accessor method, after binding `self`.

    ## Notes
    - Current version binds to the namespace accessor
        - That makes sense in the context of this being a "method"
    - The accessor is only being kept around for typing
        - which isn't working great anyway
    - The wrapper functions would be identical if `unary_accessor.__get__` did
        - `return MethodType(self._wrapper_method, instance.compliant)`
    """

    def __call__(
        self, node: IR, frame: DeprecatedFrameT_contra, name: str, /
    ) -> ColumnT_co: ...


# NOTE: Equivalent to writing a sub-protocol
# runtime requires that `FunctionExpr` is not a ForwardRef
FunctionImplMethod = ExprMethod[
    Self_, ir.FunctionExpr[F_contra], DeprecatedFrameT_contra, ColumnT_co
]
BoundFunctionImplMethod = BoundExprMethod[
    ir.FunctionExpr[F_contra], DeprecatedFrameT_contra, ColumnT_co
]
