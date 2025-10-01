from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan.compliant.dataframe import (
        CompliantBaseFrame,
        CompliantDataFrame,
        EagerDataFrame,
    )
    from narwhals._plan.compliant.group_by import GroupByResolver
    from narwhals._plan.compliant.namespace import CompliantNamespace
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.protocols import (
        CompliantExpr,
        CompliantScalar,
        EagerExpr,
        EagerScalar,
        ExprDispatch,
        LazyExpr,
        LazyScalar,
    )
    from narwhals._utils import Version

T = TypeVar("T")
R_co = TypeVar("R_co", covariant=True)
LengthT = TypeVar("LengthT")
NativeT_co = TypeVar("NativeT_co", covariant=True, default=Any)

ConcatT1 = TypeVar("ConcatT1")
ConcatT2 = TypeVar("ConcatT2", default=ConcatT1)

ColumnT = TypeVar("ColumnT")
ColumnT_co = TypeVar("ColumnT_co", covariant=True)

ResolverT_co = TypeVar("ResolverT_co", bound="GroupByResolver", covariant=True)

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
