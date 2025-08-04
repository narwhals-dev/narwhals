from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, Protocol

from narwhals._compliant.expr import NativeExpr

if TYPE_CHECKING:
    from narwhals._sql.dataframe import SQLLazyFrame
    from narwhals._sql.expr import SQLExpr
    from narwhals.dtypes import Boolean
    from typing_extensions import Self

    # TODO: check we 
    SQLExprAny = SQLExpr[Any, Any]
    SQLLazyFrameAny = SQLLazyFrame[Any, Any, Any]

SQLExprT = TypeVar("SQLExprT", bound="SQLExprAny")
SQLExprT_contra = TypeVar("SQLExprT_contra", bound="SQLExprAny", contravariant=True)
SQLLazyFrameT = TypeVar("SQLLazyFrameT", bound="SQLLazyFrameAny")
NativeSQLExprT = TypeVar("NativeSQLExprT", bound="NativeSQLExpr") 

class NativeSQLExpr(NativeExpr, Protocol):
    # both Self because we're comparing an expression with an expression? 
    def __gt__(self, value: Any, /) -> Self: ...

    def __lt__(self, value: Any, /) -> Self: ...

    def __ge__(self, value: Any, /) -> Self: ...

    # def __le__(self, value: Self) -> Self: ...

    # def __eq__(self, value: Self) -> Self: ...

    # def __ne__(self, value: Self) -> Self: ...
    # # do we want any more of the arithmetic methods? I wasn't sure between lefthan & righthand methods..
    # def __sub__(self, value: Self) -> Self: ...

    def __add__(self, value: Any, /) -> Self: ...

    def __sub__(self, value: Any, /) -> Self: ...

    def __truediv__(self, value: Any, /) -> Self: ...

    # def __mul__(self, value: Self) -> Self: ...

    #def __invert__(self) -> Self: ...



