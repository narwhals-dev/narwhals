from __future__ import annotations

from typing import TYPE_CHECKING

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
            output_names = expr._evaluate_output_names(df)
            aliases = (
                output_names
                if expr._alias_output_names is None
                else expr._alias_output_names(output_names)
            )
            if len(output_names) > 1:
                # For multi-output aggregations, e.g. `df.group_by('a').agg(nw.all().mean())`, we skip
                # the keys, else they would appear duplicated in the output.
                output_names, aliases = zip(
                    *[
                        (x, alias)
                        for x, alias in zip(output_names, aliases)
                        if x not in self._keys
                    ]
                )
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
