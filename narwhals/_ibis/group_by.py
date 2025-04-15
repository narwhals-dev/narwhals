from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Sequence

from narwhals._compliant import LazyGroupBy

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from typing_extensions import Self

    from narwhals._ibis.dataframe import IbisLazyFrame
    from narwhals._ibis.expr import IbisExpr


class IbisGroupBy(LazyGroupBy["IbisLazyFrame", "IbisExpr", "ir.Expr"]):
    def __init__(
        self: Self,
        df: IbisLazyFrame,
        keys: Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        self._compliant_frame = df.drop_nulls(subset=None) if drop_null_keys else df
        self._keys: list[str] = list(keys)

    def _alias_native_expr(self, native_expr: ir.Expr, alias: str) -> ir.Expr:
        return native_expr.name(alias)

    def agg(self: Self, *exprs: IbisExpr) -> IbisLazyFrame:
        agg_columns = list(self._evaluate_exprs(exprs))
        return self.compliant._with_native(
            self.compliant.native.group_by(self._keys).aggregate(*agg_columns)
        )
