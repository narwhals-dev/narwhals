from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

from narwhals.exceptions import AnonymousExprError

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

    def agg(
        self: Self,
        *exprs: DuckDBExpr,
    ) -> DuckDBLazyFrame:
        output_names: list[str] = copy(self._keys)
        for expr in exprs:
            if expr._output_names is None:  # pragma: no cover
                msg = "group_by.agg"
                raise AnonymousExprError.from_expr_name(msg)

            output_names.extend(expr._output_names)

        agg_columns = [
            *self._keys,
            *(x for expr in exprs for x in expr(self._compliant_frame)),
        ]
        return self._compliant_frame._from_native_frame(
            self._compliant_frame._native_frame.aggregate(
                agg_columns, group_expr=",".join(f'"{key}"' for key in self._keys)
            )
        )
