from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._polars.namespace import PolarsNamespace
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import extract_native
from narwhals._polars.utils import reverse_translate_dtype
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType


class PolarsExpr:
    def __init__(self, expr: Any) -> None:
        self._native_expr = expr
        self._implementation = Implementation.POLARS

    def __repr__(self) -> str:  # pragma: no cover
        return "PolarsExpr"

    def __narwhals_expr__(self) -> Self:  # pragma: no cover
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:  # pragma: no cover
        return PolarsNamespace(backend_version=self._backend_version)

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

    def __eq__(self, other: object) -> Self:  # type: ignore[override]
        return self._from_native_expr(self._native_expr.__eq__(extract_native(other)))

    def __ne__(self, other: object) -> Self:  # type: ignore[override]
        return self._from_native_expr(self._native_expr.__ne__(extract_native(other)))

    def __ge__(self, other: Any) -> Self:
        return self._from_native_expr(self._native_expr.__ge__(extract_native(other)))

    def __gt__(self, other: Any) -> Self:
        return self._from_native_expr(self._native_expr.__gt__(extract_native(other)))

    def __le__(self, other: Any) -> Self:
        return self._from_native_expr(self._native_expr.__le__(extract_native(other)))

    def __lt__(self, other: Any) -> Self:
        return self._from_native_expr(self._native_expr.__lt__(extract_native(other)))

    def __and__(self, other: PolarsExpr | bool | Any) -> Self:
        return self._from_native_expr(self._native_expr.__and__(extract_native(other)))

    def __or__(self, other: PolarsExpr | bool | Any) -> Self:
        return self._from_native_expr(self._native_expr.__or__(extract_native(other)))

    def __add__(self, other: Any) -> Self:
        return self._from_native_expr(self._native_expr.__add__(extract_native(other)))

    def __radd__(self, other: Any) -> Self:
        return self._from_native_expr(self._native_expr.__radd__(extract_native(other)))

    def __sub__(self, other: Any) -> Self:
        return self._from_native_expr(self._native_expr.__sub__(extract_native(other)))

    def __rsub__(self, other: Any) -> Self:
        return self._from_native_expr(self._native_expr.__rsub__(extract_native(other)))

    def __mul__(self, other: Any) -> Self:
        return self._from_native_expr(self._native_expr.__mul__(extract_native(other)))

    def __rmul__(self, other: Any) -> Self:
        return self._from_native_expr(self._native_expr.__rmul__(extract_native(other)))

    def __pow__(self, other: Any) -> Self:
        return self._from_native_expr(self._native_expr.__pow__(extract_native(other)))

    def __rpow__(self, other: Any) -> Self:
        return self._from_native_expr(self._native_expr.__rpow__(extract_native(other)))

    def __invert__(self) -> Self:
        return self._from_native_expr(self._native_expr.__invert__())

    @property
    def dt(self) -> PolarsExprDateTimeNamespace:
        return PolarsExprDateTimeNamespace(self)

    @property
    def str(self) -> PolarsExprStringNamespace:
        return PolarsExprStringNamespace(self)

    @property
    def cat(self) -> PolarsExprCatNamespace:
        return PolarsExprCatNamespace(self)

    @property
    def name(self: Self) -> PolarsExprNameNamespace:
        return PolarsExprNameNamespace(self)


class PolarsExprDateTimeNamespace:
    def __init__(self, expr: PolarsExpr) -> None:
        self._expr = expr

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._expr._from_native_expr(
                getattr(self._expr._native_expr.dt, attr)(*args, **kwargs)
            )

        return func


class PolarsExprStringNamespace:
    def __init__(self, expr: PolarsExpr) -> None:
        self._expr = expr

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._expr._from_native_expr(
                getattr(self._expr._native_expr.str, attr)(*args, **kwargs)
            )

        return func


class PolarsExprCatNamespace:
    def __init__(self, expr: PolarsExpr) -> None:
        self._expr = expr

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._expr._from_native_expr(
                getattr(self._expr._native_expr.cat, attr)(*args, **kwargs)
            )

        return func


class PolarsExprNameNamespace:
    def __init__(self, expr: PolarsExpr) -> None:
        self._expr = expr

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._expr._from_native_expr(
                getattr(self._expr._native_expr.name, attr)(*args, **kwargs)
            )

        return func
