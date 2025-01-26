from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._expression_parsing import evaluate_output_names_and_aliases

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBGroupBy:
    def __init__(
        self: Self,
        compliant_frame: DuckDBLazyFrame,
        keys: list[str],
        drop_null_keys: bool,  # noqa: FBT001
    ) -> None:
        if drop_null_keys:
            self._compliant_frame = compliant_frame.drop_nulls(subset=None)
        else:
            self._compliant_frame = compliant_frame
        self._keys = keys

    def agg(self: Self, *exprs: DuckDBExpr) -> DuckDBLazyFrame:
        agg_columns = self._keys.copy()
        df = self._compliant_frame
        for expr in exprs:
            _, aliases = evaluate_output_names_and_aliases(expr, df, self._keys)
            native_expressions = expr(df)
            agg_columns.extend(
                [
                    native_expression.alias(alias)
                    for native_expression, alias in zip(native_expressions, aliases)
                ]
            )

        return self._compliant_frame._from_native_frame(
            self._compliant_frame._native_frame.aggregate(
                agg_columns, group_expr=",".join(f'"{key}"' for key in self._keys)
            )
        )
