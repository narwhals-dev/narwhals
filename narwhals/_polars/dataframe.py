from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import cast
from typing import overload

import polars as pl

from narwhals._polars.namespace import PolarsNamespace
from narwhals._polars.series import PolarsSeries
from narwhals._polars.utils import catch_polars_exception
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import native_to_narwhals_dtype
from narwhals.dependencies import is_numpy_array_1d
from narwhals.exceptions import ColumnNotFoundError
from narwhals.utils import Implementation
from narwhals.utils import _into_arrow_table
from narwhals.utils import convert_str_slice_to_int_slice
from narwhals.utils import is_compliant_series
from narwhals.utils import is_index_selector
from narwhals.utils import is_sequence_like
from narwhals.utils import is_sequence_like_ints
from narwhals.utils import is_slice_none
from narwhals.utils import parse_columns_to_drop
from narwhals.utils import parse_version
from narwhals.utils import requires
from narwhals.utils import validate_backend_version

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Callable
    from typing import TypeVar

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self
    from typing_extensions import TypeAlias
    from typing_extensions import TypeIs

    from narwhals._polars.group_by import PolarsGroupBy
    from narwhals._polars.group_by import PolarsLazyGroupBy
    from narwhals._translate import IntoArrowTable
    from narwhals.dtypes import DType
    from narwhals.schema import Schema
    from narwhals.typing import CompliantDataFrame
    from narwhals.typing import CompliantLazyFrame
    from narwhals.typing import JoinStrategy
    from narwhals.typing import PivotAgg
    from narwhals.typing import _2DArray
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

    T = TypeVar("T")
    R = TypeVar("R")

Method: TypeAlias = "Callable[..., R]"
"""Generic alias representing all methods implemented via `__getattr__`.

Where `R` is the return type.
"""


class PolarsDataFrame:
    clone: Method[Self]
    collect: Method[CompliantDataFrame[Any, Any, Any]]
    drop_nulls: Method[Self]
    estimated_size: Method[int | float]
    explode: Method[Self]
    filter: Method[Self]
    gather_every: Method[Self]
    item: Method[Any]
    iter_rows: Method[Iterator[tuple[Any, ...]] | Iterator[Mapping[str, Any]]]
    is_unique: Method[PolarsSeries]
    join_asof: Method[Self]
    rename: Method[Self]
    row: Method[tuple[Any, ...]]
    rows: Method[Sequence[tuple[Any, ...]] | Sequence[Mapping[str, Any]]]
    sample: Method[Self]
    select: Method[Self]
    sort: Method[Self]
    to_arrow: Method[pa.Table]
    to_pandas: Method[pd.DataFrame]
    unique: Method[Self]
    with_columns: Method[Self]
    # NOTE: `write_csv` requires an `@overload` for `str | None`
    # Can't do that here 😟
    write_csv: Method[Any]
    write_parquet: Method[None]

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._native_frame = df
        self._backend_version = backend_version
        self._implementation = Implementation.POLARS
        self._version = version
        validate_backend_version(self._implementation, self._backend_version)

    @classmethod
    def from_arrow(cls, data: IntoArrowTable, /, *, context: _FullContext) -> Self:
        if context._backend_version >= (1, 3):
            native = pl.DataFrame(data)
        else:
            native = cast("pl.DataFrame", pl.from_arrow(_into_arrow_table(data, context)))
        return cls.from_native(native, context=context)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        /,
        *,
        context: _FullContext,
        schema: Mapping[str, DType] | Schema | None,
    ) -> Self:
        from narwhals.schema import Schema

        pl_schema = Schema(schema).to_polars() if schema is not None else schema
        return cls.from_native(pl.from_dict(data, pl_schema), context=context)

    @staticmethod
    def _is_native(obj: pl.DataFrame | Any) -> TypeIs[pl.DataFrame]:
        return isinstance(obj, pl.DataFrame)

    @classmethod
    def from_native(cls, data: pl.DataFrame, /, *, context: _FullContext) -> Self:
        return cls(
            data, backend_version=context._backend_version, version=context._version
        )

    @classmethod
    def from_numpy(
        cls,
        data: _2DArray,
        /,
        *,
        context: _FullContext,  # NOTE: Maybe only `Implementation`?
        schema: Mapping[str, DType] | Schema | Sequence[str] | None,
    ) -> Self:
        from narwhals.schema import Schema

        pl_schema = (
            Schema(schema).to_polars()
            if isinstance(schema, (Mapping, Schema))
            else schema
        )
        return cls.from_native(pl.from_numpy(data, pl_schema), context=context)

    @property
    def native(self) -> pl.DataFrame:
        return self._native_frame

    def __repr__(self) -> str:  # pragma: no cover
        return "PolarsDataFrame"

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def __native_namespace__(self) -> ModuleType:
        if self._implementation is Implementation.POLARS:
            return self._implementation.to_native_namespace()

        msg = f"Expected polars, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def _with_version(self, version: Version) -> Self:
        return self.__class__(
            self.native, backend_version=self._backend_version, version=version
        )

    def _with_native(self, df: pl.DataFrame) -> Self:
        return self.__class__(
            df, backend_version=self._backend_version, version=self._version
        )

    @overload
    def _from_native_object(self, obj: pl.Series) -> PolarsSeries: ...

    @overload
    def _from_native_object(self, obj: pl.DataFrame) -> Self: ...

    @overload
    def _from_native_object(self, obj: T) -> T: ...

    def _from_native_object(
        self, obj: pl.Series | pl.DataFrame | T
    ) -> Self | PolarsSeries | T:
        if isinstance(obj, pl.Series):
            return PolarsSeries.from_native(obj, context=self)
        if self._is_native(obj):
            return self._with_native(obj)
        # scalar
        return obj

    def __len__(self) -> int:
        return len(self.native)

    def head(self, n: int) -> Self:
        return self._with_native(self.native.head(n))

    def tail(self, n: int) -> Self:
        return self._with_native(self.native.tail(n))

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            pos, kwds = extract_args_kwargs(args, kwargs)
            try:
                return self._from_native_object(getattr(self.native, attr)(*pos, **kwds))
            except pl.exceptions.ColumnNotFoundError as e:  # pragma: no cover
                msg = f"{e!s}\n\nHint: Did you mean one of these columns: {self.columns}?"
                raise ColumnNotFoundError(msg) from e
            except Exception as e:  # noqa: BLE001
                raise catch_polars_exception(e, self._backend_version) from None

        return func

    def __array__(
        self, dtype: Any | None = None, *, copy: bool | None = None
    ) -> _2DArray:
        if self._backend_version < (0, 20, 28) and copy is not None:
            msg = "`copy` in `__array__` is only supported for Polars>=0.20.28"
            raise NotImplementedError(msg)
        if self._backend_version < (0, 20, 28):
            return self.native.__array__(dtype)
        return self.native.__array__(dtype)

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _2DArray:
        return self.native.to_numpy()

    def collect_schema(self) -> dict[str, DType]:
        if self._backend_version < (1,):
            return {
                name: native_to_narwhals_dtype(
                    dtype, self._version, self._backend_version
                )
                for name, dtype in self.native.schema.items()
            }
        else:
            collected_schema = self.native.collect_schema()
            return {
                name: native_to_narwhals_dtype(
                    dtype, self._version, self._backend_version
                )
                for name, dtype in collected_schema.items()
            }

    @property
    def shape(self) -> tuple[int, int]:
        return self.native.shape

    def __getitem__(self, item: Any) -> Any:
        rows, columns = item
        if is_compliant_series(rows):
            rows = rows.native
        if is_compliant_series(columns):
            columns = columns.native
        if self._backend_version > (0, 20, 30):
            return self._from_native_object(self.native.__getitem__((rows, columns)))
        else:  # pragma: no cover
            # TODO(marco): we can delete this branch after Polars==0.20.30 becomes the minimum
            # Polars version we support
            rows = list(rows) if isinstance(rows, tuple) else rows
            columns = list(columns) if isinstance(columns, tuple) else columns
            if is_numpy_array_1d(columns):
                columns = columns.tolist()

            is_int_col_indexer = is_index_selector(columns)
            native = self.native
            if not is_slice_none(columns):
                if hasattr(columns, "__len__") and len(columns) == 0:
                    native = native.select()
                if is_int_col_indexer and not isinstance(columns, (slice, range)):
                    native = native[:, columns]
                elif is_int_col_indexer and isinstance(columns, (slice, range)):
                    native = native.select(
                        self.columns[slice(columns.start, columns.stop, columns.step)]
                    )
                elif isinstance(columns, (slice, range)):
                    native = native.select(
                        self.columns[
                            slice(*convert_str_slice_to_int_slice(columns, self.columns))
                        ]
                    )
                elif is_int_col_indexer:
                    native = native[:, columns]
                elif is_sequence_like(columns):
                    native = native.select(columns)
                else:
                    msg = "Unreachable code"
                    raise AssertionError(msg)

            if not is_slice_none(rows):
                if isinstance(rows, int):
                    native = native[[rows], :]
                elif (
                    isinstance(rows, (slice, range))
                    or is_sequence_like_ints(rows)
                    or isinstance(rows, pl.Series)
                ):
                    native = native[rows, :]
                else:
                    msg = "Unreachable code"
                    raise AssertionError(msg)

            return self._with_native(native)

    def simple_select(self, *column_names: str) -> Self:
        return self._with_native(self.native.select(*column_names))

    def aggregate(self, *exprs: Any) -> Self:
        return self.select(*exprs)

    def get_column(self, name: str) -> PolarsSeries:
        return PolarsSeries.from_native(self.native.get_column(name), context=self)

    def iter_columns(self) -> Iterator[PolarsSeries]:
        for series in self.native.iter_columns():
            yield PolarsSeries.from_native(series, context=self)

    @property
    def columns(self) -> list[str]:
        return self.native.columns

    @property
    def schema(self) -> dict[str, DType]:
        return {
            name: native_to_narwhals_dtype(dtype, self._version, self._backend_version)
            for name, dtype in self.native.schema.items()
        }

    def lazy(
        self, *, backend: Implementation | None = None
    ) -> CompliantLazyFrame[Any, Any]:
        if backend is None or backend is Implementation.POLARS:
            return PolarsLazyFrame.from_native(self.native.lazy(), context=self)
        elif backend is Implementation.DUCKDB:
            import duckdb  # ignore-banned-import

            from narwhals._duckdb.dataframe import DuckDBLazyFrame

            # NOTE: (F841) is a false positive
            df = self.native  # noqa: F841
            return DuckDBLazyFrame(
                duckdb.table("df"),
                backend_version=parse_version(duckdb),
                version=self._version,
            )
        elif backend is Implementation.DASK:
            import dask  # ignore-banned-import
            import dask.dataframe as dd  # ignore-banned-import

            from narwhals._dask.dataframe import DaskLazyFrame

            return DaskLazyFrame(
                dd.from_pandas(self.native.to_pandas()),
                backend_version=parse_version(dask),
                version=self._version,
            )
        raise AssertionError  # pragma: no cover

    @overload
    def to_dict(self, *, as_series: Literal[True]) -> dict[str, PolarsSeries]: ...

    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...

    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, PolarsSeries] | dict[str, list[Any]]:
        if as_series:
            return {
                name: PolarsSeries.from_native(col, context=self)
                for name, col in self.native.to_dict().items()
            }
        else:
            return self.native.to_dict(as_series=False)

    def group_by(self, *keys: str, drop_null_keys: bool) -> PolarsGroupBy:
        from narwhals._polars.group_by import PolarsGroupBy

        return PolarsGroupBy(self, keys, drop_null_keys=drop_null_keys)

    def with_row_index(self, name: str) -> Self:
        if self._backend_version < (0, 20, 4):
            return self._with_native(self.native.with_row_count(name))
        return self._with_native(self.native.with_row_index(name))

    def drop(self, columns: Sequence[str], *, strict: bool) -> Self:
        to_drop = parse_columns_to_drop(
            compliant_frame=self, columns=columns, strict=strict
        )
        return self._with_native(self.native.drop(to_drop))

    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self:
        if self._backend_version < (1, 0, 0):
            return self._with_native(
                self.native.melt(
                    id_vars=index,
                    value_vars=on,
                    variable_name=variable_name,
                    value_name=value_name,
                )
            )
        return self._with_native(
            self.native.unpivot(
                on=on, index=index, variable_name=variable_name, value_name=value_name
            )
        )

    @requires.backend_version((1,))
    def pivot(
        self,
        on: Sequence[str],
        *,
        index: Sequence[str] | None,
        values: Sequence[str] | None,
        aggregate_function: PivotAgg | None,
        sort_columns: bool,
        separator: str,
    ) -> Self:
        try:
            result = self.native.pivot(
                on,
                index=index,
                values=values,
                aggregate_function=aggregate_function,
                sort_columns=sort_columns,
                separator=separator,
            )
        except Exception as e:  # noqa: BLE001
            raise catch_polars_exception(e, self._backend_version) from None
        return self._from_native_object(result)

    def to_polars(self) -> pl.DataFrame:
        return self.native

    def join(
        self,
        other: Self,
        *,
        how: JoinStrategy,
        left_on: Sequence[str] | None,
        right_on: Sequence[str] | None,
        suffix: str,
    ) -> Self:
        how_native = (
            "outer" if (self._backend_version < (0, 20, 29) and how == "full") else how
        )
        try:
            return self._with_native(
                self.native.join(
                    other=other.native,
                    how=how_native,  # type: ignore[arg-type]
                    left_on=left_on,
                    right_on=right_on,
                    suffix=suffix,
                )
            )
        except Exception as e:  # noqa: BLE001
            raise catch_polars_exception(e, self._backend_version) from None


class PolarsLazyFrame:
    drop_nulls: Method[Self]
    explode: Method[Self]
    filter: Method[Self]
    gather_every: Method[Self]
    head: Method[Self]
    join_asof: Method[Self]
    rename: Method[Self]
    select: Method[Self]
    sort: Method[Self]
    tail: Method[Self]
    unique: Method[Self]
    with_columns: Method[Self]
    # NOTE: Temporary, just trying to factor out utils
    _evaluate_expr: Any

    def __init__(
        self,
        df: pl.LazyFrame,
        *,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._native_frame = df
        self._backend_version = backend_version
        self._implementation = Implementation.POLARS
        self._version = version
        validate_backend_version(self._implementation, self._backend_version)

    @staticmethod
    def _is_native(obj: pl.LazyFrame | Any) -> TypeIs[pl.LazyFrame]:
        return isinstance(obj, pl.LazyFrame)

    @classmethod
    def from_native(cls, data: pl.LazyFrame, /, *, context: _FullContext) -> Self:
        return cls(
            data, backend_version=context._backend_version, version=context._version
        )

    def __repr__(self) -> str:  # pragma: no cover
        return "PolarsLazyFrame"

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def __native_namespace__(self) -> ModuleType:
        if self._implementation is Implementation.POLARS:
            return self._implementation.to_native_namespace()

        msg = f"Expected polars, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def _with_native(self, df: pl.LazyFrame) -> Self:
        return self.__class__(
            df, backend_version=self._backend_version, version=self._version
        )

    def _with_version(self, version: Version) -> Self:
        return self.__class__(
            self.native, backend_version=self._backend_version, version=version
        )

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            pos, kwds = extract_args_kwargs(args, kwargs)
            try:
                return self._with_native(getattr(self.native, attr)(*pos, **kwds))
            except pl.exceptions.ColumnNotFoundError as e:  # pragma: no cover
                raise ColumnNotFoundError(str(e)) from e

        return func

    def _iter_columns(self) -> Iterator[PolarsSeries]:  # pragma: no cover
        yield from self.collect(self._implementation).iter_columns()

    @property
    def native(self) -> pl.LazyFrame:
        return self._native_frame

    @property
    def columns(self) -> list[str]:
        return self.native.columns

    @property
    def schema(self) -> dict[str, DType]:
        schema = self.native.schema
        return {
            name: native_to_narwhals_dtype(dtype, self._version, self._backend_version)
            for name, dtype in schema.items()
        }

    def collect_schema(self) -> dict[str, DType]:
        if self._backend_version < (1,):
            return {
                name: native_to_narwhals_dtype(
                    dtype, self._version, self._backend_version
                )
                for name, dtype in self.native.schema.items()
            }
        else:
            try:
                collected_schema = self.native.collect_schema()
            except Exception as e:  # noqa: BLE001
                raise catch_polars_exception(e, self._backend_version) from None
            return {
                name: native_to_narwhals_dtype(
                    dtype, self._version, self._backend_version
                )
                for name, dtype in collected_schema.items()
            }

    def collect(
        self,
        backend: Implementation | None,
        **kwargs: Any,
    ) -> CompliantDataFrame[Any, Any, Any]:
        try:
            result = self.native.collect(**kwargs)
        except Exception as e:  # noqa: BLE001
            raise catch_polars_exception(e, self._backend_version) from None

        if backend is None or backend is Implementation.POLARS:
            return PolarsDataFrame.from_native(result, context=self)

        if backend is Implementation.PANDAS:
            import pandas as pd  # ignore-banned-import

            from narwhals._pandas_like.dataframe import PandasLikeDataFrame

            return PandasLikeDataFrame(
                result.to_pandas(),
                implementation=Implementation.PANDAS,
                backend_version=parse_version(pd),
                version=self._version,
                validate_column_names=False,
            )

        if backend is Implementation.PYARROW:
            import pyarrow as pa  # ignore-banned-import

            from narwhals._arrow.dataframe import ArrowDataFrame

            return ArrowDataFrame(
                result.to_arrow(),
                backend_version=parse_version(pa),
                version=self._version,
                validate_column_names=False,
            )

        msg = f"Unsupported `backend` value: {backend}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def group_by(self, *keys: str, drop_null_keys: bool) -> PolarsLazyGroupBy:
        from narwhals._polars.group_by import PolarsLazyGroupBy

        return PolarsLazyGroupBy(self, keys, drop_null_keys=drop_null_keys)

    def with_row_index(self, name: str) -> Self:
        if self._backend_version < (0, 20, 4):
            return self._with_native(self.native.with_row_count(name))
        return self._with_native(self.native.with_row_index(name))

    def drop(self, columns: Sequence[str], *, strict: bool) -> Self:
        if self._backend_version < (1, 0, 0):
            return self._with_native(self.native.drop(columns))
        return self._with_native(self.native.drop(columns, strict=strict))

    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self:
        if self._backend_version < (1, 0, 0):
            return self._with_native(
                self.native.melt(
                    id_vars=index,
                    value_vars=on,
                    variable_name=variable_name,
                    value_name=value_name,
                )
            )
        return self._with_native(
            self.native.unpivot(
                on=on, index=index, variable_name=variable_name, value_name=value_name
            )
        )

    def simple_select(self, *column_names: str) -> Self:
        return self._with_native(self.native.select(*column_names))

    def aggregate(self, *exprs: Any) -> Self:
        return self.select(*exprs)

    def join(
        self,
        other: Self,
        *,
        how: JoinStrategy,
        left_on: Sequence[str] | None,
        right_on: Sequence[str] | None,
        suffix: str,
    ) -> Self:
        how_native = (
            "outer" if (self._backend_version < (0, 20, 29) and how == "full") else how
        )
        return self._with_native(
            self.native.join(
                other=other.native,
                how=how_native,  # type: ignore[arg-type]
                left_on=left_on,
                right_on=right_on,
                suffix=suffix,
            )
        )
