from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.dtypes import DType
from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    from narwhals._sqlalchemy.dataframe import SQLAlchemyLazyFrame


def maybe_evaluate(df: SQLAlchemyLazyFrame, obj: Any) -> Any:
    from narwhals._sqlalchemy.expr import SQLAlchemyExpr

    if isinstance(obj, SQLAlchemyExpr):
        column_results = obj._call(df)
        if len(column_results) != 1:  # pragma: no cover
            msg = "Multi-output expressions (e.g. `nw.all()` or `nw.col('a', 'b')`) not supported in this context"
            raise NotImplementedError(msg)
        column_result = column_results[0]
        if obj._returns_scalar:
            msg = "Reductions are not yet supported for DuckDB, at least until they implement duckdb.WindowExpression"
            raise NotImplementedError(msg)
        return column_result
    if isinstance_or_issubclass(obj, DType):
        return obj
    return obj
