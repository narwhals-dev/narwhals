from __future__ import annotations
def validate_comparand(lhs, rhs, df):
    from narwhals._duckdb.expr import DuckDBExpr
    import duckdb
    if isinstance(rhs, DuckDBExpr):
        res = rhs._call(df)
        assert len(res) == 1
        return res
    return duckdb.ConstantExpression(rhs)

def get_column_name(df: SparkLikeLazyFrame, column: Column) -> str:
    return str(df._native_frame.select(column).columns[0])

def maybe_evaluate(df: SparkLikeLazyFrame, obj: Any) -> Any:
    from narwhals._duckdb.expr import DuckDBExpr
    import duckdb

    if isinstance(obj, DuckDBExpr):
        column_results = obj._call(df)
        if len(column_results) != 1:  # pragma: no cover
            msg = "Multi-output expressions (e.g. `nw.all()` or `nw.col('a', 'b')`) not supported in this context"
            raise NotImplementedError(msg)
        column_result = column_results[0]
        if obj._returns_scalar:
            # Return scalar, let PySpark do its broadcasting
            1/0
            # return column_result.over(Window.partitionBy(F.lit(1)))
        return column_result
    return duckdb.ConstantExpression(obj)

