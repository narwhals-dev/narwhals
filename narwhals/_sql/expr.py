# In here, we could make SQLExpr
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from narwhals._compliant.typing import AliasNames, CompliantLazyFrameT, NativeExprT
from narwhals._typing_compat import Protocol38
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self, TypeIs

    from narwhals._compliant.typing import AliasNames, WindowFunction

from narwhals._compliant.expr import LazyExpr
from narwhals._compliant.typing import WindowFunction


class SQLExpr(
    LazyExpr[CompliantLazyFrameT, NativeExprT],
    Protocol38[CompliantLazyFrameT, NativeExprT],
):
    @property
    def window_function(self) -> WindowFunction[CompliantLazyFrameT, NativeExprT]: ...

    @classmethod
    def _is_expr(cls, obj: Self | Any) -> TypeIs[Self]:
        return hasattr(obj, "__narwhals_expr__")

    def _with_callable(self, call: Callable[..., Any], /) -> Self: ...
    def _with_alias_output_names(self, func: AliasNames | None, /) -> Self: ...

    @property
    def _backend_version(self) -> tuple[int, ...]:
        return self._implementation._backend_version()

    @classmethod
    def _alias_native(cls, expr: NativeExprT, name: str, /) -> NativeExprT: ...

    @classmethod
    def _from_elementwise_horizontal_op(
        cls, func: Callable[[Iterable[NativeExprT]], NativeExprT], *exprs: Self
    ) -> Self: ...

    def _with_binary(self, op: Callable[..., NativeExprT], other: Self | Any) -> Self: ...

    def __eq__(self, other: Self) -> Self:  # type: ignore[override]
        return self._with_binary(lambda expr, other: expr.__eq__(other), other)

    def __ne__(self, other: Self) -> Self:  # type: ignore[override]
        return self._with_binary(lambda expr, other: expr.__ne__(other), other)

    def __add__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__add__(other), other)

    def __sub__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__sub__(other), other)

    def __rsub__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: other - expr, other).alias("literal")

    def __mul__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__mul__(other), other)

    def __truediv__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__truediv__(other), other)

    def __rtruediv__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: other / expr, other).alias("literal")

    def __floordiv__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__floordiv__(other), other)

    def __rfloordiv__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: other // expr, other).alias(
            "literal"
        )

    def __pow__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__pow__(other), other)

    def __rpow__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: other**expr, other).alias("literal")

    def __mod__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__mod__(other), other)

    def __rmod__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: other % expr, other).alias("literal")

    def __ge__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__ge__(other), other)

    def __gt__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__gt__(other), other)

    def __le__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__le__(other), other)

    def __lt__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__lt__(other), other)

    def __and__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__and__(other), other)

    def __or__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__or__(other), other)

    arg_max: not_implemented = not_implemented()
    arg_min: not_implemented = not_implemented()
    arg_true: not_implemented = not_implemented()
    ewm_mean: not_implemented = not_implemented()
    gather_every: not_implemented = not_implemented()
    head: not_implemented = not_implemented()
    map_batches: not_implemented = not_implemented()
    mode: not_implemented = not_implemented()
    replace_strict: not_implemented = not_implemented()
    sort: not_implemented = not_implemented()
    sample: not_implemented = not_implemented()
    tail: not_implemented = not_implemented()

    # namespaces
    cat: not_implemented = not_implemented()  # type: ignore[assignment]
