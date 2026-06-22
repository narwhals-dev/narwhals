from __future__ import annotations

from typing import TYPE_CHECKING, overload

from narwhals._plan import _guards, expressions as ir
from narwhals._plan.exceptions import literal_type_error
from narwhals.dtypes import Object

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr
    from narwhals._plan.series import Series
    from narwhals._plan.typing import NativeSeriesT
    from narwhals.typing import IntoDType, PythonLiteral


@overload
def lit(
    value: PythonLiteral | Series[NativeSeriesT], dtype: IntoDType | None = None
) -> Expr: ...
@overload
def lit(value: object, dtype: Object | type[Object]) -> Expr: ...
def lit(
    value: PythonLiteral | Series[NativeSeriesT] | object, dtype: IntoDType | None = None
) -> Expr:
    if _guards.is_series(value):
        return ir.lit_series(value).to_narwhals()
    if not _guards.is_python_literal(value) and not (
        isinstance(dtype, Object) or dtype is Object
    ):
        raise literal_type_error(value)
    # NOTE: Don't want to support this in `ir.Lit`, so the ignore will do for now...
    # Ideally there would be a different `ExprIR` that is only opted-in to by `polars` and `pandas`
    return ir.lit(value, dtype).to_narwhals()  # type: ignore[type-var]
