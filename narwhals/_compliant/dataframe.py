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

from narwhals._compliant.typing import CompliantExprT_contra
from narwhals._compliant.typing import CompliantSeriesT
from narwhals._compliant.typing import EagerExprT_contra
from narwhals._compliant.typing import EagerSeriesT
from narwhals._compliant.typing import NativeFrameT_co
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals.utils import Version
from narwhals.utils import _StoresNative
from narwhals.utils import deprecated

if TYPE_CHECKING:
    from io import BytesIO
    from pathlib import Path

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self
    from typing_extensions import TypeAlias

    from narwhals._compliant.group_by import CompliantGroupBy
    from narwhals.dtypes import DType
    from narwhals.typing import SizeUnit
    from narwhals.typing import _2DArray
    from narwhals.utils import Implementation

    Incomplete: TypeAlias = Any

__all__ = ["CompliantDataFrame", "CompliantLazyFrame", "EagerDataFrame"]

T = TypeVar("T")


class CompliantDataFrame(
    _StoresNative[NativeFrameT_co],
    Sized,
    Protocol[CompliantSeriesT, CompliantExprT_contra, NativeFrameT_co],
):
    _native_frame: Any
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    def __narwhals_dataframe__(self) -> Self: ...
    def __narwhals_namespace__(self) -> Any: ...
    def __array__(self, dtype: Any, *, copy: bool | None) -> _2DArray: ...
    def __getitem__(self, item: Any) -> CompliantSeriesT | Self: ...
    def simple_select(self, *column_names: str) -> Self:
        """`select` where all args are column names."""
        ...

    def aggregate(self, *exprs: CompliantExprT_contra) -> Self:
        """`select` where all args are aggregations or literals.

        (so, no broadcasting is necessary).
        """
        # NOTE: Ignore is to avoid an intermittent false positive
        return self.select(*exprs)  # pyright: ignore[reportArgumentType]

    @property
    def native(self) -> NativeFrameT_co:
        return self._native_frame  # type: ignore[no-any-return]

    @property
    def columns(self) -> Sequence[str]: ...
    @property
    def schema(self) -> Mapping[str, DType]: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    def clone(self) -> Self: ...
    def collect(
        self, backend: Implementation | None, **kwargs: Any
    ) -> CompliantDataFrame[Any, Any, Any]: ...
    def collect_schema(self) -> Mapping[str, DType]: ...
    def drop(self, columns: Sequence[str], *, strict: bool) -> Self: ...
    def drop_nulls(self, subset: Sequence[str] | None) -> Self: ...
    def estimated_size(self, unit: SizeUnit) -> int | float: ...
    def explode(self: Self, columns: Sequence[str]) -> Self: ...
    def filter(self, predicate: CompliantExprT_contra | Incomplete) -> Self: ...
    def gather_every(self, n: int, offset: int) -> Self: ...
    def get_column(self, name: str) -> CompliantSeriesT: ...
    def group_by(
        self, *keys: str, drop_null_keys: bool
    ) -> CompliantGroupBy[Self, Any]: ...
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
        how: Literal["inner", "left", "full", "cross", "semi", "anti"],
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
    def lazy(self, *, backend: Implementation | None) -> CompliantLazyFrame[Any, Any]: ...
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
    def select(self, *exprs: CompliantExprT_contra) -> Self: ...
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
        maintain_order: bool | None = None,
    ) -> Self: ...
    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self: ...
    def with_columns(self, *exprs: CompliantExprT_contra) -> Self: ...
    def with_row_index(self, name: str) -> Self: ...
    @overload
    def write_csv(self, file: None) -> str: ...
    @overload
    def write_csv(self, file: str | Path | BytesIO) -> None: ...
    def write_csv(self, file: str | Path | BytesIO | None) -> str | None: ...
    def write_parquet(self, file: str | Path | BytesIO) -> None: ...


class CompliantLazyFrame(
    _StoresNative[NativeFrameT_co], Protocol[CompliantExprT_contra, NativeFrameT_co]
):
    _native_frame: Any
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    def __narwhals_lazyframe__(self) -> Self: ...
    def __narwhals_namespace__(self) -> Any: ...

    def simple_select(self, *column_names: str) -> Self:
        """`select` where all args are column names."""
        ...

    def aggregate(self, *exprs: CompliantExprT_contra) -> Self:
        """`select` where all args are aggregations or literals.

        (so, no broadcasting is necessary).
        """
        ...

    def _change_version(self, version: Version) -> Self: ...

    @property
    def native(self) -> NativeFrameT_co:
        return self._native_frame  # type: ignore[no-any-return]

    @property
    def columns(self) -> Sequence[str]: ...
    @property
    def schema(self) -> Mapping[str, DType]: ...
    def _iter_columns(self) -> Iterator[Any]: ...
    def collect(
        self, backend: Implementation | None, **kwargs: Any
    ) -> CompliantDataFrame[Any, Any, Any]: ...
    def collect_schema(self) -> Mapping[str, DType]: ...
    def drop(self, columns: Sequence[str], *, strict: bool) -> Self: ...
    def drop_nulls(self, subset: Sequence[str] | None) -> Self: ...
    def explode(self: Self, columns: Sequence[str]) -> Self: ...
    def filter(self, predicate: CompliantExprT_contra | Incomplete) -> Self: ...
    @deprecated(
        "`LazyFrame.gather_every` is deprecated and will be removed in a future version."
    )
    def gather_every(self, n: int, offset: int) -> Self: ...
    def group_by(
        self, *keys: str, drop_null_keys: bool
    ) -> CompliantGroupBy[Self, Any]: ...
    def head(self, n: int) -> Self: ...
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
    def rename(self, mapping: Mapping[str, str]) -> Self: ...
    def select(self, *exprs: CompliantExprT_contra) -> Self: ...
    def sort(
        self, *by: str, descending: bool | Sequence[bool], nulls_last: bool
    ) -> Self: ...
    @deprecated("`LazyFrame.tail` is deprecated and will be removed in a future version.")
    def tail(self, n: int) -> Self: ...
    def unique(
        self,
        subset: Sequence[str] | None,
        *,
        keep: Literal["any", "none"],
    ) -> Self: ...
    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self: ...
    def with_columns(self, *exprs: CompliantExprT_contra) -> Self: ...
    def with_row_index(self, name: str) -> Self: ...
    def _evaluate_expr(self, expr: CompliantExprT_contra, /) -> Any:
        result = expr(self)
        assert len(result) == 1  # debug assertion  # noqa: S101
        return result[0]


class EagerDataFrame(
    CompliantDataFrame[EagerSeriesT, EagerExprT_contra, NativeFrameT_co],
    CompliantLazyFrame[EagerExprT_contra, NativeFrameT_co],
    Protocol[EagerSeriesT, EagerExprT_contra, NativeFrameT_co],
):
    def _evaluate_expr(self, expr: EagerExprT_contra, /) -> EagerSeriesT:
        """Evaluate `expr` and ensure it has a **single** output."""
        result: Sequence[EagerSeriesT] = expr(self)
        assert len(result) == 1  # debug assertion  # noqa: S101
        return result[0]

    def _evaluate_into_exprs(self, *exprs: EagerExprT_contra) -> Sequence[EagerSeriesT]:
        return list(chain.from_iterable(self._evaluate_into_expr(expr) for expr in exprs))

    def _evaluate_into_expr(self, expr: EagerExprT_contra, /) -> Sequence[EagerSeriesT]:
        """Return list of raw columns.

        For eager backends we alias operations at each step.

        As a safety precaution, here we can check that the expected result names match those
        we were expecting from the various `evaluate_output_names` / `alias_output_names` calls.

        Note that for PySpark / DuckDB, we are less free to liberally set aliases whenever we want.
        """
        _, aliases = evaluate_output_names_and_aliases(expr, self, [])
        result = expr(self)
        if list(aliases) != (result_aliases := [s.name for s in result]):
            msg = f"Safety assertion failed, expected {aliases}, got {result_aliases}"
            raise AssertionError(msg)
        return result

    def _extract_comparand(self, other: EagerSeriesT, /) -> Any:
        """Extract native Series, broadcasting to `len(self)` if necessary."""
        ...
