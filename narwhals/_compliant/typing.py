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
    from narwhals._compliant.expr import EagerExpr
    from narwhals._compliant.expr import NativeExpr
    from narwhals._compliant.namespace import EagerNamespace
    from narwhals._compliant.series import CompliantSeries
    from narwhals._compliant.series import EagerSeries
    from narwhals.typing import NativeFrame

__all__ = [
    "AliasName",
    "AliasNames",
    "CompliantDataFrameT",
    "CompliantFrameT",
    "CompliantLazyFrameT",
    "CompliantSeriesT",
    "IntoCompliantExpr",
    "NativeFrameT_co",
]
NativeExprT_co = TypeVar("NativeExprT_co", bound="NativeExpr", covariant=True)
CompliantSeriesT = TypeVar("CompliantSeriesT", bound="CompliantSeries")
CompliantSeriesOrNativeExprT_co = TypeVar(
    "CompliantSeriesOrNativeExprT_co",
    bound="CompliantSeries | NativeExpr",
    covariant=True,
)
NativeFrameT_co = TypeVar("NativeFrameT_co", bound="NativeFrame", covariant=True)
CompliantFrameT = TypeVar(
    "CompliantFrameT",
    bound="CompliantDataFrame[Any, Any, Any] | CompliantLazyFrame[Any, Any]",
)
CompliantDataFrameT = TypeVar(
    "CompliantDataFrameT", bound="CompliantDataFrame[Any, Any, Any]"
)
CompliantLazyFrameT = TypeVar("CompliantLazyFrameT", bound="CompliantLazyFrame[Any, Any]")
IntoCompliantExpr: TypeAlias = "CompliantExpr[CompliantFrameT, CompliantSeriesOrNativeExprT_co] | CompliantSeriesOrNativeExprT_co"
CompliantExprT = TypeVar("CompliantExprT", bound="CompliantExpr[Any, Any]")
CompliantExprT_contra = TypeVar(
    "CompliantExprT_contra", bound="CompliantExpr[Any, Any]", contravariant=True
)

EagerDataFrameT = TypeVar("EagerDataFrameT", bound="EagerDataFrame[Any, Any, Any]")
EagerSeriesT = TypeVar("EagerSeriesT", bound="EagerSeries[Any]")
EagerSeriesT_co = TypeVar("EagerSeriesT_co", bound="EagerSeries[Any]", covariant=True)
EagerExprT = TypeVar("EagerExprT", bound="EagerExpr[Any, Any]")
EagerExprT_contra = TypeVar(
    "EagerExprT_contra", bound="EagerExpr[Any, Any]", contravariant=True
)
EagerNamespaceAny: TypeAlias = (
    "EagerNamespace[EagerDataFrame[Any, Any, Any], EagerSeries[Any], EagerExpr[Any, Any]]"
)

AliasNames: TypeAlias = Callable[[Sequence[str]], Sequence[str]]
AliasName: TypeAlias = Callable[[str], str]
