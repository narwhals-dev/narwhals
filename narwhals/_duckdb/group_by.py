from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Sequence

from narwhals._compliant import CompliantGroupBy

if TYPE_CHECKING:
    from duckdb import Expression
    from typing_extensions import Self

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.expr import DuckDBExpr


# NOTE: No depth-tracking
class DuckDBGroupBy(CompliantGroupBy["DuckDBLazyFrame", "DuckDBExpr"]):
    def __init__(
        self: Self,
        df: DuckDBLazyFrame,
        keys: Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        self._compliant_frame = df.drop_nulls(subset=None) if drop_null_keys else df
        self._keys = list(keys)

    def agg(self: Self, *exprs: DuckDBExpr) -> DuckDBLazyFrame:
        agg_columns: list[str | Expression] = list(self._keys)
        for expr in exprs:
            output_names = expr._evaluate_output_names(self.compliant)
            aliases = (
                output_names
                if expr._alias_output_names is None
                else expr._alias_output_names(output_names)
            )
            native_expressions = expr(self.compliant)
            exclude = (
                self._keys
                if expr._function_name.split("->", maxsplit=1)[0] in {"all", "selector"}
                else []
            )
            agg_columns.extend(
                native_expression.alias(alias)
                for native_expression, output_name, alias in zip(
                    native_expressions, output_names, aliases
                )
                if output_name not in exclude
            )
        return self.compliant._from_native_frame(
            self.compliant.native.aggregate(agg_columns)  # type: ignore[arg-type]
        )
