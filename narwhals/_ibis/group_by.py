from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from narwhals._compliant import LazyGroupBy

if TYPE_CHECKING:
    import ibis.expr.types as ir  # noqa: F401

    from narwhals._ibis.dataframe import IbisLazyFrame
    from narwhals._ibis.expr import IbisExpr


class IbisGroupBy(LazyGroupBy["IbisLazyFrame", "IbisExpr", "ir.Value"]):
    def __init__(
        self,
        df: IbisLazyFrame,
        keys: Sequence[str] | Sequence[IbisExpr],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        frame, self._keys, self._output_key_names = self._parse_keys(df, keys=keys)
        self._compliant_frame = frame.drop_nulls(self._keys) if drop_null_keys else frame

    def agg(self, *exprs: IbisExpr) -> IbisLazyFrame:
        native = self.compliant.native
        return self.compliant._with_native(
            native.group_by(self._keys).aggregate(*self._evaluate_exprs(exprs))
        ).rename(dict(zip(self._keys, self._output_key_names)))
