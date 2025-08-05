from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from narwhals._compliant.expr import NativeExpr

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._sql.dataframe import SQLLazyFrame
    from narwhals._sql.expr import SQLExpr

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

    def __le__(self, value: Any, /) -> Self: ...

    def __eq__(self, value: Any, /) -> Self: ...

    def __ne__(self, value: Any, /) -> Self: ...

    def __add__(self, value: Any, /) -> Self: ...

    def __sub__(self, value: Any, /) -> Self: ...

    def __truediv__(self, value: Any, /) -> Self: ...

    def __mul__(self, value: Any, /) -> Self: ...

    def __invert__(self) -> Self: ...
