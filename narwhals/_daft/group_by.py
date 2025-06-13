from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from narwhals._compliant import LazyGroupBy

if TYPE_CHECKING:
    from daft import Expression  # noqa: F401

    from narwhals._daft.dataframe import DaftLazyFrame
    from narwhals._daft.expr import DaftExpr


class DaftGroupBy(LazyGroupBy["DaftLazyFrame", "DaftExpr", "Expression"]):
    def __init__(
        self,
        df: DaftLazyFrame,
        keys: Sequence[DaftExpr] | Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        frame, self._keys, self._output_key_names = self._parse_keys(df, keys=keys)
        self._compliant_frame = frame.drop_nulls(self._keys) if drop_null_keys else frame

    def agg(self, *exprs: DaftExpr) -> DaftLazyFrame:
        result = (
            self.compliant.native.groupby(*self._keys).agg(*agg_columns)
            if (agg_columns := list(self._evaluate_exprs(exprs)))
            else self.compliant.native.select(*self._keys).unique()
        )

        return self.compliant._with_native(result).rename(
            dict(zip(self._keys, self._output_key_names))
        )
