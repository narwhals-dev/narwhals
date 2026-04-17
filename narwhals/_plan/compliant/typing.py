from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan.compliant.column import ExprDispatch
    from narwhals._plan.compliant.dataframe import (
        CompliantDataFrame,
        CompliantFrame,
        EagerDataFrame,
    )
    from narwhals._plan.compliant.expr import CompliantExpr, EagerExpr
    from narwhals._plan.compliant.group_by import GroupByResolver
    from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
    from narwhals._plan.compliant.namespace import CompliantNamespace
    from narwhals._plan.compliant.scalar import CompliantScalar, EagerScalar
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._utils import Version

R_co = TypeVar("R_co", covariant=True)
LengthT = TypeVar("LengthT")
ResolverT_co = TypeVar("ResolverT_co", bound="GroupByResolver", covariant=True)
Native = TypeVar("Native")
"""Unbounded type variable, representing *any* native object.

Assume nothing, permit anything; rely on well-defined protocols to do the talking.
"""
FromNative = TypeVar("FromNative")
"""Same as `Native`, but should be scoped to constructor method(s) and not the class."""

ExprAny: TypeAlias = "CompliantExpr[Any]"
ScalarAny: TypeAlias = "CompliantScalar[Any]"
SeriesAny: TypeAlias = "CompliantSeries[Any]"
FrameAny: TypeAlias = "CompliantFrame[Any]"
DataFrameAny: TypeAlias = "CompliantDataFrame[Any, Any]"
LazyFrameAny: TypeAlias = "CompliantLazyFrame[Any]"
NamespaceAny: TypeAlias = "CompliantNamespace[Any, Any, Any]"

EagerExprAny: TypeAlias = "EagerExpr[Any, Any]"
EagerScalarAny: TypeAlias = "EagerScalar[Any, Any]"
EagerDataFrameAny: TypeAlias = "EagerDataFrame[Any, Any, Any]"

ExprT = TypeVar("ExprT", bound=ExprAny)
ExprT_co = TypeVar("ExprT_co", bound=ExprAny, covariant=True)
ScalarT_co = TypeVar("ScalarT_co", bound=ScalarAny, covariant=True)
"""TODO @dangotbanned: Investigate using `ExprT_co` as a default.

Could also/alternatively use `bound=ExprAny`.
"""

SeriesT = TypeVar("SeriesT", bound=SeriesAny)
SeriesT_co = TypeVar("SeriesT_co", bound=SeriesAny, covariant=True)
FrameT = TypeVar("FrameT", bound=FrameAny)
FrameT_co = TypeVar("FrameT_co", bound=FrameAny, covariant=True)
FrameT_contra = TypeVar("FrameT_contra", bound=FrameAny, contravariant=True)
DataFrameT = TypeVar("DataFrameT", bound=DataFrameAny)
DataFrameT_co = TypeVar("DataFrameT_co", bound=DataFrameAny, covariant=True)
LazyFrameT = TypeVar("LazyFrameT", bound=LazyFrameAny)
LazyFrameT_co = TypeVar("LazyFrameT_co", bound=LazyFrameAny, covariant=True)
LazyFrameT_contra = TypeVar("LazyFrameT_contra", bound=LazyFrameAny, contravariant=True)
NamespaceT_co = TypeVar(
    "NamespaceT_co", bound="NamespaceAny", covariant=True, default="NamespaceAny"
)

EagerExprT_co = TypeVar("EagerExprT_co", bound=EagerExprAny, covariant=True)
EagerScalarT_co = TypeVar("EagerScalarT_co", bound=EagerScalarAny, covariant=True)
EagerDataFrameT = TypeVar("EagerDataFrameT", bound=EagerDataFrameAny)


Ctx: TypeAlias = "ExprDispatch[FrameT_contra, R_co]"
"""Type of an unknown expression dispatch context.

- `FrameT_contra`: Compliant data/lazyframe
- `R_co`: Upper bound return type of the context
"""


class SupportsNarwhalsNamespace(Protocol[NamespaceT_co]):
    def __narwhals_namespace__(self) -> NamespaceT_co: ...


# TODO @dangotbanned: `_version` (attribute) is not implemented, but typed in a `HasVersion` as `Version`
# - `version` (property) is implemented there, defined as referencing `self._version`
# - I remember this being an issue for pyarrow typing
# NOTE: Most of the `Compliant*` protocols are using this detail to provide constructors that assign to it
# - Would be better if we didn't need to pass this around everywhere
# - Really the version should be scoped to all operations
#  - the property would reference that new place
class HasVersion(Protocol):
    _version: Version

    # NOTE: Unlike `nw._utils._StoresVersion`, here the property is public
    @property
    def version(self) -> Version:
        """Narwhals API version (V1 or MAIN)."""
        return self._version
