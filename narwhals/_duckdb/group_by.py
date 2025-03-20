from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING
from typing import Sequence

from narwhals._compliant import LazyGroupBy

if TYPE_CHECKING:
    from duckdb import Expression  # noqa: F401
    from typing_extensions import Self

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBGroupBy(LazyGroupBy["DuckDBLazyFrame", "DuckDBExpr", "Expression"]):
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
        agg_columns = list(chain(self._keys, self._evaluate_exprs(exprs)))
        return self.compliant._from_native_frame(
            self.compliant.native.aggregate(agg_columns)  # type: ignore[arg-type]
        )
