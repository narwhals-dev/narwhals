from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._compliant.dataframe import CompliantDataFrame
    from narwhals._compliant.dataframe import CompliantLazyFrame
    from narwhals._compliant.dataframe import EagerDataFrame
    from narwhals._compliant.expr import CompliantExpr
    from narwhals._compliant.expr import DepthTrackingExpr
    from narwhals._compliant.expr import EagerExpr
    from narwhals._compliant.expr import LazyExpr
    from narwhals._compliant.expr import NativeExpr
    from narwhals._compliant.namespace import EagerNamespace
    from narwhals._compliant.series import CompliantSeries
    from narwhals._compliant.series import EagerSeries
    from narwhals.typing import NativeFrame
    from narwhals.typing import NativeSeries

__all__ = [
    "AliasName",
    "AliasNames",
    "CompliantDataFrameT",
    "CompliantFrameT",
    "CompliantLazyFrameT",
    "CompliantSeriesT",
    "IntoCompliantExpr",
    "NativeFrameT_co",
    "NativeSeriesT_co",
]
CompliantExprAny: TypeAlias = "CompliantExpr[Any, Any]"
CompliantSeriesAny: TypeAlias = "CompliantSeries[Any]"
CompliantSeriesOrNativeExprAny: TypeAlias = "CompliantSeriesAny | NativeExpr"
CompliantDataFrameAny: TypeAlias = "CompliantDataFrame[Any, Any, Any]"
CompliantLazyFrameAny: TypeAlias = "CompliantLazyFrame[Any, Any]"
CompliantFrameAny: TypeAlias = "CompliantDataFrameAny | CompliantLazyFrameAny"

DepthTrackingExprAny: TypeAlias = "DepthTrackingExpr[Any, Any]"

EagerDataFrameAny: TypeAlias = "EagerDataFrame[Any, Any, Any]"
EagerSeriesAny: TypeAlias = "EagerSeries[Any]"
EagerExprAny: TypeAlias = "EagerExpr[Any, Any]"
EagerNamespaceAny: TypeAlias = (
    "EagerNamespace[EagerDataFrameAny, EagerSeriesAny, EagerExprAny]"
)

LazyExprAny: TypeAlias = "LazyExpr[Any, Any]"

NativeExprT = TypeVar("NativeExprT", bound="NativeExpr")
NativeExprT_co = TypeVar("NativeExprT_co", bound="NativeExpr", covariant=True)
NativeSeriesT = TypeVar("NativeSeriesT", bound="NativeSeries")
NativeSeriesT_co = TypeVar("NativeSeriesT_co", bound="NativeSeries", covariant=True)
NativeFrameT_co = TypeVar("NativeFrameT_co", bound="NativeFrame", covariant=True)

CompliantExprT = TypeVar("CompliantExprT", bound=CompliantExprAny)
CompliantExprT_contra = TypeVar(
    "CompliantExprT_contra", bound=CompliantExprAny, contravariant=True
)
CompliantSeriesT = TypeVar("CompliantSeriesT", bound=CompliantSeriesAny)
CompliantSeriesOrNativeExprT = TypeVar(
    "CompliantSeriesOrNativeExprT", bound=CompliantSeriesOrNativeExprAny
)
CompliantSeriesOrNativeExprT_co = TypeVar(
    "CompliantSeriesOrNativeExprT_co",
    bound=CompliantSeriesOrNativeExprAny,
    covariant=True,
)
CompliantFrameT = TypeVar("CompliantFrameT", bound=CompliantFrameAny)
CompliantFrameT_co = TypeVar(
    "CompliantFrameT_co", bound=CompliantFrameAny, covariant=True
)
CompliantDataFrameT = TypeVar("CompliantDataFrameT", bound=CompliantDataFrameAny)
CompliantDataFrameT_co = TypeVar(
    "CompliantDataFrameT_co", bound=CompliantDataFrameAny, covariant=True
)
CompliantLazyFrameT = TypeVar("CompliantLazyFrameT", bound=CompliantLazyFrameAny)
CompliantLazyFrameT_co = TypeVar(
    "CompliantLazyFrameT_co", bound=CompliantLazyFrameAny, covariant=True
)
IntoCompliantExpr: TypeAlias = "CompliantExpr[CompliantFrameT, CompliantSeriesOrNativeExprT_co] | CompliantSeriesOrNativeExprT_co"

DepthTrackingExprT = TypeVar("DepthTrackingExprT", bound=DepthTrackingExprAny)
DepthTrackingExprT_contra = TypeVar(
    "DepthTrackingExprT_contra", bound=DepthTrackingExprAny, contravariant=True
)

EagerExprT = TypeVar("EagerExprT", bound=EagerExprAny)
EagerExprT_contra = TypeVar("EagerExprT_contra", bound=EagerExprAny, contravariant=True)
EagerSeriesT = TypeVar("EagerSeriesT", bound=EagerSeriesAny)
EagerSeriesT_co = TypeVar("EagerSeriesT_co", bound=EagerSeriesAny, covariant=True)

# NOTE: `pyright` gives false (8) positives if this uses `EagerDataFrameAny`?
EagerDataFrameT = TypeVar("EagerDataFrameT", bound="EagerDataFrame[Any, Any, Any]")

LazyExprT_contra = TypeVar("LazyExprT_contra", bound=LazyExprAny, contravariant=True)

AliasNames: TypeAlias = Callable[[Sequence[str]], Sequence[str]]
AliasName: TypeAlias = Callable[[str], str]
