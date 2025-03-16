from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Literal
from typing import Mapping
from typing import Protocol
from typing import Sequence
from typing import Sized
from typing import TypeVar
from typing import overload

from narwhals._compliant.typing import CompliantSeriesT
from narwhals._compliant.typing import EagerSeriesT
from narwhals._expression_parsing import evaluate_output_names_and_aliases

if TYPE_CHECKING:
    from io import BytesIO
    from pathlib import Path

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self
    from typing_extensions import TypeIs

    from narwhals._compliant.expr import EagerExpr
    from narwhals.dtypes import DType
    from narwhals.typing import SizeUnit
    from narwhals.typing import _2DArray
    from narwhals.utils import Implementation

__all__ = ["CompliantDataFrame", "CompliantLazyFrame", "EagerDataFrame"]

T = TypeVar("T")


class CompliantDataFrame(Sized, Protocol[CompliantSeriesT]):
    def __narwhals_dataframe__(self) -> Self: ...
    def __narwhals_namespace__(self) -> Any: ...
    def __array__(self, dtype: Any, *, copy: bool | None) -> _2DArray: ...
    def __getitem__(self, item: Any) -> CompliantSeriesT | Self: ...
    def simple_select(self, *column_names: str) -> Self:
        """`select` where all args are column names."""
        ...

    # NOTE: Can we remove this now?
    # `DaskLazyFrame` is the only one not the same as `select`
    def aggregate(self, *exprs: Any) -> Self:  # pragma: no cover
        """`select` where all args are aggregations or literals.

        (so, no broadcasting is necessary).
        """
        ...

    @property
    def columns(self) -> Sequence[str]: ...
    @property
    def schema(self) -> Mapping[str, DType]: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    def clone(self) -> Self: ...
    def collect(
        self, backend: Implementation | None, **kwargs: Any
    ) -> CompliantDataFrame[Any]: ...
    def collect_schema(self) -> Mapping[str, DType]: ...
    def drop(self, columns: Sequence[str], strict: bool) -> Self: ...  # noqa: FBT001
    def drop_nulls(self, subset: Sequence[str] | None) -> Self: ...
    def estimated_size(self, unit: SizeUnit) -> int | float: ...
    def filter(self, predicate: Any) -> Self: ...
    def gather_every(self, n: int, offset: int) -> Self: ...
    def get_column(self, name: str) -> CompliantSeriesT: ...
    def group_by(self, *keys: str, drop_null_keys: bool) -> Any: ...
    def head(self, n: int) -> Self: ...
    def item(self, row: int | None, column: int | str | None) -> Any: ...
    def iter_columns(self) -> Iterator[CompliantSeriesT]: ...
    def iter_rows(
        self, *, named: bool, buffer_size: int
    ) -> Iterator[tuple[Any, ...]] | Iterator[Mapping[str, Any]]: ...
    def is_unique(self) -> CompliantSeriesT: ...
    def join(
        self: Self,
        other: Self,
        *,
        how: Literal["left", "inner", "cross", "anti", "semi"],
        left_on: Sequence[str] | None,
        right_on: Sequence[str] | None,
        suffix: str,
    ) -> Self: ...
    def join_asof(
        self: Self,
        other: Self,
        *,
        left_on: str | None,
        right_on: str | None,
        by_left: Sequence[str] | None,
        by_right: Sequence[str] | None,
        strategy: Literal["backward", "forward", "nearest"],
        suffix: str,
    ) -> Self: ...
    def lazy(self, *, backend: Implementation | None) -> CompliantLazyFrame: ...
    def rename(self, mapping: Mapping[str, str]) -> Self: ...
    def row(self, index: int) -> tuple[Any, ...]: ...
    def rows(
        self, *, named: bool
    ) -> Sequence[tuple[Any, ...]] | Sequence[Mapping[str, Any]]: ...
    def sample(
        self,
        n: int | None,
        *,
        fraction: float | None,
        with_replacement: bool,
        seed: int | None,
    ) -> Self: ...
    def select(self, *exprs: Any) -> Self: ...
    def sort(
        self, *by: str, descending: bool | Sequence[bool], nulls_last: bool
    ) -> Self: ...
    def tail(self, n: int) -> Self: ...
    def to_arrow(self) -> pa.Table: ...
    def to_numpy(self) -> _2DArray: ...
    def to_pandas(self) -> pd.DataFrame: ...
    def to_polars(self) -> pl.DataFrame: ...
    @overload
    def to_dict(self, *, as_series: Literal[True]) -> dict[str, CompliantSeriesT]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, CompliantSeriesT] | dict[str, list[Any]]: ...
    def unique(
        self,
        subset: Sequence[str] | None,
        *,
        keep: Literal["any", "first", "last", "none"],
        maintain_order: bool | None,
    ) -> Self: ...
    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self: ...
    def with_columns(self, *exprs: Any) -> Self: ...
    def with_row_index(self, name: str) -> Self: ...
    @overload
    def write_csv(self, file: None) -> str: ...
    @overload
    def write_csv(self, file: str | Path | BytesIO) -> None: ...
    def write_csv(self, file: str | Path | BytesIO | None) -> str | None: ...
    def write_parquet(self, file: str | Path | BytesIO) -> None: ...


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
