from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING
from typing import Sequence

from narwhals._compliant import LazyGroupBy

if TYPE_CHECKING:
    from duckdb import Expression  # noqa: F401

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBGroupBy(LazyGroupBy["DuckDBLazyFrame", "DuckDBExpr", "Expression"]):
    def __init__(
        self, df: DuckDBLazyFrame, keys: Sequence[DuckDBExpr], /, *, drop_null_keys: bool
    ) -> None:
        super().__init__(df, keys, drop_null_keys=drop_null_keys)
        self._compliant_frame = (
            self.compliant.drop_nulls(subset=self._keys)
            if drop_null_keys
            else self.compliant
        )

    def agg(self, *exprs: DuckDBExpr) -> DuckDBLazyFrame:
        agg_columns = list(chain(self._keys, self._evaluate_exprs(exprs)))
        return self.compliant._with_native(
            self.compliant.native.aggregate(agg_columns)  # type: ignore[arg-type]
        ).rename(dict(zip(self._keys, self._output_key_names)))
