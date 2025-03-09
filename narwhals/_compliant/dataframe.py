from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Mapping
from typing import Protocol
from typing import Sequence
from typing import TypeVar

from narwhals._compliant.typing import CompliantSeriesT_co
from narwhals._compliant.typing import EagerSeriesT
from narwhals._compliant.typing import EagerSeriesT_co

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing_extensions import TypeIs

    from narwhals._compliant.expr import EagerExpr
    from narwhals.dtypes import DType

__all__ = ["CompliantDataFrame", "CompliantLazyFrame", "EagerDataFrame"]

T = TypeVar("T")


class CompliantDataFrame(Protocol[CompliantSeriesT_co]):
    def __narwhals_dataframe__(self) -> Self: ...
    def __narwhals_namespace__(self) -> Any: ...
    def simple_select(
        self, *column_names: str
    ) -> Self: ...  # `select` where all args are column names.
    def aggregate(self, *exprs: Any) -> Self:  # pragma: no cover
        ...  # `select` where all args are aggregations or literals
        # (so, no broadcasting is necessary).

    @property
    def columns(self) -> Sequence[str]: ...
    @property
    def schema(self) -> Mapping[str, DType]: ...
    def get_column(self, name: str) -> CompliantSeriesT_co: ...
    def iter_columns(self) -> Iterator[CompliantSeriesT_co]: ...


class CompliantLazyFrame(Protocol):
    def __narwhals_lazyframe__(self) -> Self: ...
    def __narwhals_namespace__(self) -> Any: ...
    def simple_select(
        self, *column_names: str
    ) -> Self: ...  # `select` where all args are column names.
    def aggregate(self, *exprs: Any) -> Self:  # pragma: no cover
        ...  # `select` where all args are aggregations or literals
        # (so, no broadcasting is necessary).

    @property
    def columns(self) -> Sequence[str]: ...
    @property
    def schema(self) -> Mapping[str, DType]: ...
    def _iter_columns(self) -> Iterator[Any]: ...


class EagerDataFrame(CompliantDataFrame[EagerSeriesT_co], Protocol[EagerSeriesT_co]):
    def _maybe_evaluate_expr(
        self, expr: EagerExpr[Self, EagerSeriesT_co] | T, /
    ) -> EagerSeriesT_co | T:
        if is_eager_expr(expr):
            result: Sequence[EagerSeriesT_co] = expr(self)
            if len(result) > 1:
                msg = (
                    "Multi-output expressions (e.g. `nw.all()` or `nw.col('a', 'b')`) "
                    "are not supported in this context"
                )
                raise ValueError(msg)
            return result[0]
        return expr


# NOTE: `mypy` is requiring the gymnastics here and is very fragile
# DON'T CHANGE THIS or `EagerDataFrame._maybe_evaluate_expr`
def is_eager_expr(
    obj: EagerExpr[Any, EagerSeriesT] | Any,
) -> TypeIs[EagerExpr[Any, EagerSeriesT]]:
    return hasattr(obj, "__narwhals_expr__")
