from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant.expr import CompliantExprNameNamespace
from narwhals._compliant.expr import LazyExprNamespace

if TYPE_CHECKING:
    from narwhals._compliant.typing import AliasName
    from narwhals._dask.expr import DaskExpr


class DaskExprNameNamespace(
    LazyExprNamespace["DaskExpr"], CompliantExprNameNamespace["DaskExpr"]
):
    def _from_callable(self, func: AliasName, /, *, alias: bool = True) -> DaskExpr:
        expr = self.compliant
        return type(expr)(
            call=expr._call,
            depth=expr._depth,
            function_name=expr._function_name,
            evaluate_output_names=expr._evaluate_output_names,
            alias_output_names=self._alias_output_names(func) if alias else None,
            backend_version=expr._backend_version,
            version=expr._version,
            call_kwargs=expr._call_kwargs,
        )
