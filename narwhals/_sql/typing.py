from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from narwhals._sql.dataframe import SQLLazyFrame
    from narwhals._sql.expr import SQLExpr

    SQLExprAny = SQLExpr[Any, Any]
    SQLLazyFrameAny = SQLLazyFrame[Any, Any, Any]

SQLExprT = TypeVar("SQLExprT", bound="SQLExprAny")
SQLExprT_contra = TypeVar("SQLExprT_contra", bound="SQLExprAny", contravariant=True)
SQLLazyFrameT = TypeVar("SQLLazyFrameT", bound="SQLLazyFrameAny")


# it needs to inherit from SQLExpr, but getting errors if passing SQLExprT. I have not sorted out what relation 
# (covarinant, contravariant or intravariant it needs to be yet) it needs to have to its parent class
class NativeSQLExpr(SQLExprT):
    # not sure how to initialise the class, I want it to have all the methods etc of SQLExprT and add its own operator
    # methods in addition to those 
    def __init__(self, other: [Int | Float]) -> None:
        super().__init__(other)

    # I've seen lots of examples of the operator dunder methods being used, but I'm struggling with 
    # the typing. for sql, there are nice binary examples, but that's not the right method
    # e.g.
    def __gt__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__gt__(other), other)
    
    def __and__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__and__(other), other)
    
    # so it should probably look more like something from the polars examples,
    # where we just apply the native method: 
    def __gt__(self, other: Any) -> Self:
        return self._with_native(self.native.__gt__(extract_native(other)))

    def __le__(self, other: Any) -> Self:
        return self._with_native(self.native.__le__(extract_native(other)))

    
       

    


