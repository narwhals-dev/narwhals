from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType
from narwhals._polars.namespace import PolarsNamespace
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import reverse_translate_dtype


class PolarsExpr:
    def __init__(self, expr: Any) -> None:
        self._native_expr = expr

    def __repr__(self) -> str:
        return "PolarsExpr"

    def __narwhals_expr__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace()

    def _from_native_expr(self, expr: Any) -> Self:
        return self.__class__(expr)

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._from_native_expr(
                getattr(self._native_expr, attr)(*args, **kwargs)
            )

        return func

    def cast(self, dtype: DType) -> Self:
        expr = self._native_expr
        dtype = reverse_translate_dtype(dtype)
        return self._from_native_expr(expr.cast(dtype))
