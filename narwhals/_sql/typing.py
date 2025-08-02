from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from narwhals._compliant.expr import NativeExpr

if TYPE_CHECKING:
    from narwhals._sql.dataframe import SQLLazyFrame
    from narwhals._sql.expr import SQLExpr
    from narwhals.dtypes import Boolean

    SQLExprAny = SQLExpr[Any, Any]
    SQLLazyFrameAny = SQLLazyFrame[Any, Any, Any]

SQLExprT = TypeVar("SQLExprT", bound="SQLExprAny")
SQLExprT_contra = TypeVar("SQLExprT_contra", bound="SQLExprAny", contravariant=True)
SQLLazyFrameT = TypeVar("SQLLazyFrameT", bound="SQLLazyFrameAny")


class NativeSQLExpr(NativeExpr):
    # TODO @mp: fix input type for all these!
    def __gt__(self, value: float) -> Boolean: ...

    def __lt__(self, value: float) -> Boolean: ...

    def __ge__(self, value: float) -> Boolean: ...

    def __le__(self, value: float) -> Boolean: ...

    def __eq__(self, value: float) -> Boolean: ...

    def __ne__(self, value: float) -> Boolean: ...
    # do we want any more of the arithmetic methods? I wasn't sure between lefthan & righthand methods..
    def __sub__(self, value: float) -> Any: ...

    def __add__(self, value: float) -> Any: ...

    def __truediv__(self, value: float) -> Any: ...

    def __mul__(self, value: float) -> Any: ...
