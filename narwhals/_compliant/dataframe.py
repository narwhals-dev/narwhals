from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Mapping
from typing import Protocol
from typing import Sequence
from typing import TypeVar

from narwhals._compliant.typing import CompliantSeriesT_co
from narwhals._compliant.typing import EagerSeriesT
from narwhals._expression_parsing import evaluate_output_names_and_aliases

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


class EagerDataFrame(CompliantDataFrame[EagerSeriesT], Protocol[EagerSeriesT]):
    def _maybe_evaluate_expr(
        self, expr: EagerExpr[Self, EagerSeriesT] | T, /
    ) -> EagerSeriesT | T:
        if is_eager_expr(expr):
            result: Sequence[EagerSeriesT] = expr(self)
            if len(result) > 1:
                msg = (
                    "Multi-output expressions (e.g. `nw.all()` or `nw.col('a', 'b')`) "
                    "are not supported in this context"
                )
                raise ValueError(msg)
            return result[0]
        return expr

    def _evaluate_into_exprs(
        self, *exprs: EagerExpr[Self, EagerSeriesT]
    ) -> Sequence[EagerSeriesT]:
        return list(chain.from_iterable(self._evaluate_into_expr(expr) for expr in exprs))

    def _evaluate_into_expr(
        self, expr: EagerExpr[Self, EagerSeriesT], /
    ) -> Sequence[EagerSeriesT]:
        """Return list of raw columns.

        For eager backends we alias operations at each step.

        As a safety precaution, here we can check that the expected result names match those
        we were expecting from the various `evaluate_output_names` / `alias_output_names` calls.

        Note that for PySpark / DuckDB, we are less free to liberally set aliases whenever we want.
        """
        _, aliases = evaluate_output_names_and_aliases(expr, self, [])
        result = expr(self)
        if list(aliases) != [s.name for s in result]:
            msg = f"Safety assertion failed, expected {aliases}, got {result}"
            raise AssertionError(msg)
        return result


# NOTE: `mypy` is requiring the gymnastics here and is very fragile
# DON'T CHANGE THIS or `EagerDataFrame._maybe_evaluate_expr`
def is_eager_expr(
    obj: EagerExpr[Any, EagerSeriesT] | Any,
) -> TypeIs[EagerExpr[Any, EagerSeriesT]]:
    return hasattr(obj, "__narwhals_expr__")
