from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._duckdb.utils import F, col, lit
from narwhals._sql.expr_str import SQLExprStringNamespace
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from duckdb import Expression

    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprStringNamespace(SQLExprStringNamespace["DuckDBExpr"]):
    def to_datetime(self, format: str | None) -> DuckDBExpr:
        if format is None:
            msg = "Cannot infer format with DuckDB backend, please specify `format` explicitly."
            raise NotImplementedError(msg)

        return self.compliant._with_elementwise(
            lambda expr: F("strptime", expr, lit(format))
        )

    def to_date(self, format: str | None) -> DuckDBExpr:
        if format is not None:
            return self.to_datetime(format=format).dt.date()

        compliant_expr = self.compliant
        return compliant_expr.cast(compliant_expr._version.dtypes.Date())

    def to_titlecase(self) -> DuckDBExpr:
        if (version := self.compliant._backend_version) < (1, 2):  # pragma: no cover
            msg = (
                "`Expr.str.to_titlecase` is only available in 'duckdb>=1.2', "
                f"found version {version!r}."
            )
            raise NotImplementedError(msg)
        from duckdb import LambdaExpression

        def _to_titlecase(expr: Expression) -> Expression:
            lower_expr = F("lower", expr)
            extract_expr = F(
                "regexp_extract_all", lower_expr, lit(r"[a-z0-9]*[^a-z0-9]*")
            )
            capitalize = F(
                "||",
                F("upper", F("left", col("s"), lit(1))),
                F("substr", col("s"), lit(2)),
            )
            capitalized_expr = F(
                "list_transform", extract_expr, LambdaExpression("s", capitalize)
            )
            return F("list_aggregate", capitalized_expr, lit("string_agg"), lit(""))

        return self.compliant._with_elementwise(_to_titlecase)

    replace = not_implemented()
