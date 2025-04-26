from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import cast
from typing import overload

import numpy as np

from narwhals._compliant import EagerDataFrame
from narwhals._pandas_like.series import PANDAS_TO_NUMPY_DTYPE_MISSING
from narwhals._pandas_like.series import PandasLikeSeries
from narwhals._pandas_like.utils import align_and_extract_native
from narwhals._pandas_like.utils import align_series_full_broadcast
from narwhals._pandas_like.utils import check_column_names_are_unique
from narwhals._pandas_like.utils import get_dtype_backend
from narwhals._pandas_like.utils import native_to_narwhals_dtype
from narwhals._pandas_like.utils import object_native_to_narwhals_dtype
from narwhals._pandas_like.utils import pivot_table
from narwhals._pandas_like.utils import rename
from narwhals._pandas_like.utils import select_columns_by_name
from narwhals._pandas_like.utils import set_index
from narwhals.dependencies import is_pandas_like_dataframe
from narwhals.exceptions import InvalidOperationError
from narwhals.exceptions import ShapeError
from narwhals.utils import Implementation
from narwhals.utils import _into_arrow_table
from narwhals.utils import _remap_full_join_keys
from narwhals.utils import check_column_exists
from narwhals.utils import generate_temporary_column_name
from narwhals.utils import parse_columns_to_drop
from narwhals.utils import parse_version
from narwhals.utils import scale_bytes
from narwhals.utils import validate_backend_version

if TYPE_CHECKING:
    from io import BytesIO
    from pathlib import Path
    from types import ModuleType

    import pandas as pd
    import polars as pl
    from typing_extensions import Self
    from typing_extensions import TypeAlias
    from typing_extensions import TypeIs

    from narwhals._pandas_like.expr import PandasLikeExpr
    from narwhals._pandas_like.group_by import PandasLikeGroupBy
    from narwhals._pandas_like.namespace import PandasLikeNamespace
    from narwhals._translate import IntoArrowTable
    from narwhals.dtypes import DType
    from narwhals.schema import Schema
    from narwhals.typing import AsofJoinStrategy
    from narwhals.typing import CompliantDataFrame
    from narwhals.typing import CompliantLazyFrame
    from narwhals.typing import DTypeBackend
    from narwhals.typing import JoinStrategy
    from narwhals.typing import PivotAgg
    from narwhals.typing import SizedMultiIndexSelector
    from narwhals.typing import SizedMultiNameSelector
    from narwhals.typing import SizeUnit
    from narwhals.typing import UniqueKeepStrategy
    from narwhals.typing import _2DArray
    from narwhals.typing import _SliceIndex
    from narwhals.typing import _SliceName
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

    Constructor: TypeAlias = Callable[..., pd.DataFrame]


CLASSICAL_NUMPY_DTYPES: frozenset[np.dtype[Any]] = frozenset(
    [
        np.dtype("float64"),
        np.dtype("float32"),
        np.dtype("int64"),
        np.dtype("int32"),
        np.dtype("int16"),
        np.dtype("int8"),
        np.dtype("uint64"),
        np.dtype("uint32"),
        np.dtype("uint16"),
        np.dtype("uint8"),
        np.dtype("bool"),
        np.dtype("datetime64[s]"),
        np.dtype("datetime64[ms]"),
        np.dtype("datetime64[us]"),
        np.dtype("datetime64[ns]"),
        np.dtype("timedelta64[s]"),
        np.dtype("timedelta64[ms]"),
        np.dtype("timedelta64[us]"),
        np.dtype("timedelta64[ns]"),
        np.dtype("object"),
    ]
)


class PandasLikeDataFrame(
    EagerDataFrame["PandasLikeSeries", "PandasLikeExpr", "Any", "pd.Series[Any]"]
):
    def __init__(
        self,
        native_dataframe: Any,
        *,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
        validate_column_names: bool,
    ) -> None:
        self._native_frame = native_dataframe
        self._implementation = implementation
        self._backend_version = backend_version
        self._version = version
        validate_backend_version(self._implementation, self._backend_version)
        if validate_column_names:
            check_column_names_are_unique(native_dataframe.columns)

    @classmethod
    def from_arrow(cls, data: IntoArrowTable, /, *, context: _FullContext) -> Self:
        implementation = context._implementation
        tbl = _into_arrow_table(data, context)
        if implementation.is_pandas():
            native = tbl.to_pandas()
        elif implementation.is_modin():  # pragma: no cover
            from modin.pandas.utils import from_arrow as mpd_from_arrow

            native = mpd_from_arrow(tbl)
        elif implementation.is_cudf():  # pragma: no cover
            native = implementation.to_native_namespace().DataFrame.from_arrow(tbl)
        else:  # pragma: no cover
            msg = "congratulations, you entered unreachable code - please report a bug"
            raise AssertionError(msg)
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

        implementation = context._implementation
        ns = implementation.to_native_namespace()
        Series = cast("type[pd.Series[Any]]", ns.Series)  # noqa: N806
        DataFrame = cast("type[pd.DataFrame]", ns.DataFrame)  # noqa: N806
        aligned_data: dict[str, pd.Series[Any] | Any] = {}
        left_most: PandasLikeSeries | None = None
        for name, series in data.items():
            if isinstance(series, Series):
                compliant = PandasLikeSeries.from_native(series, context=context)
                if left_most is None:
                    left_most = compliant
                    aligned_data[name] = series
                else:
                    aligned_data[name] = align_and_extract_native(left_most, compliant)[1]
            else:
                aligned_data[name] = series

        native = DataFrame.from_dict(aligned_data)
        if schema:
            it: Iterable[DTypeBackend] = (
                get_dtype_backend(dtype, implementation) for dtype in native.dtypes
            )
            native = native.astype(Schema(schema).to_pandas(it))
        return cls.from_native(native, context=context)

    @staticmethod
    def _is_native(obj: Any) -> TypeIs[Any]:
        return is_pandas_like_dataframe(obj)  # pragma: no cover

    @classmethod
    def from_native(cls, data: Any, /, *, context: _FullContext) -> Self:
        return cls(
            data,
            implementation=context._implementation,
            backend_version=context._backend_version,
            version=context._version,
            validate_column_names=True,
        )

    @classmethod
    def from_numpy(
        cls,
        data: _2DArray,
        /,
        *,
        context: _FullContext,
        schema: Mapping[str, DType] | Schema | Sequence[str] | None,
    ) -> Self:
        from narwhals.schema import Schema

        implementation = context._implementation
        DataFrame: Constructor = implementation.to_native_namespace().DataFrame  # noqa: N806
        if isinstance(schema, (Mapping, Schema)):
            it: Iterable[DTypeBackend] = (
                get_dtype_backend(native_type, implementation)
                for native_type in schema.values()
            )
            native = DataFrame(data, columns=schema.keys()).astype(
                Schema(schema).to_pandas(it)
            )
        else:
            native = DataFrame(data, columns=cls._numpy_column_names(data, schema))
        return cls.from_native(native, context=context)

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PandasLikeNamespace:
        from narwhals._pandas_like.namespace import PandasLikeNamespace

        return PandasLikeNamespace(
            self._implementation, self._backend_version, version=self._version
        )

    def __native_namespace__(self) -> ModuleType:
        if self._implementation in {
            Implementation.PANDAS,
            Implementation.MODIN,
            Implementation.CUDF,
        }:
            return self._implementation.to_native_namespace()

        msg = f"Expected pandas/modin/cudf, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def __len__(self) -> int:
        return len(self.native)

    def _with_version(self, version: Version) -> Self:
        return self.__class__(
            self.native,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=version,
            validate_column_names=False,
        )

    def _with_native(self, df: Any, *, validate_column_names: bool = True) -> Self:
        return self.__class__(
            df,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
            validate_column_names=validate_column_names,
        )

    def _extract_comparand(self, other: PandasLikeSeries) -> pd.Series[Any]:
        index = self.native.index
        if other._broadcast:
            s = other.native
            return type(s)(s.iloc[0], index=index, dtype=s.dtype, name=s.name)
        if (len_other := len(other)) != (len_idx := len(index)):
            msg = f"Expected object of length {len_idx}, got: {len_other}."
            raise ShapeError(msg)
        if other.native.index is not index:
            return set_index(
                other.native,
                index,
                implementation=other._implementation,
                backend_version=other._backend_version,
            )
        return other.native

    def get_column(self, name: str) -> PandasLikeSeries:
        return PandasLikeSeries.from_native(self.native[name], context=self)

    def __array__(self, dtype: Any = None, *, copy: bool | None = None) -> _2DArray:
        return self.to_numpy(dtype=dtype, copy=copy)

    def _gather(self, rows: SizedMultiIndexSelector[pd.Series[Any]]) -> Self:
        items = list(rows) if isinstance(rows, tuple) else rows
        return self._with_native(self.native.iloc[items, :])

    def _gather_slice(self, rows: _SliceIndex | range) -> Self:
        return self._with_native(
            self.native.iloc[slice(rows.start, rows.stop, rows.step), :],
            validate_column_names=False,
        )

    def _select_slice_name(self, columns: _SliceName) -> Self:
        start = (
            self.native.columns.get_loc(columns.start)
            if columns.start is not None
            else None
        )
        stop = (
            self.native.columns.get_loc(columns.stop) + 1
            if columns.stop is not None
            else None
        )
        selector = slice(start, stop, columns.step)
        return self._with_native(
            self.native.iloc[:, selector], validate_column_names=False
        )

    def _select_slice_index(self, columns: _SliceIndex | range) -> Self:
        return self._with_native(
            self.native.iloc[:, columns], validate_column_names=False
        )

    def _select_multi_index(
        self, columns: SizedMultiIndexSelector[pd.Series[Any]]
    ) -> Self:
        columns = list(columns) if isinstance(columns, tuple) else columns
        return self._with_native(
            self.native.iloc[:, columns], validate_column_names=False
        )

    def _select_multi_name(
        self, columns: SizedMultiNameSelector[pd.Series[Any]]
    ) -> PandasLikeDataFrame:
        return self._with_native(self.native.loc[:, columns])

    # --- properties ---
    @property
    def columns(self) -> list[str]:
        return self.native.columns.tolist()

    @overload
    def rows(
        self,
        *,
        named: Literal[True],
    ) -> list[dict[str, Any]]: ...

    @overload
    def rows(
        self,
        *,
        named: Literal[False],
    ) -> list[tuple[Any, ...]]: ...

    @overload
    def rows(
        self,
        *,
        named: bool,
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]: ...

    def rows(self, *, named: bool) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        if not named:
            # cuDF does not support itertuples. But it does support to_dict!
            if self._implementation is Implementation.CUDF:
                # Extract the row values from the named rows
                return [tuple(row.values()) for row in self.rows(named=True)]

            return list(self.native.itertuples(index=False, name=None))

        return self.native.to_dict(orient="records")

    def iter_columns(self) -> Iterator[PandasLikeSeries]:
        for _name, series in self.native.items():  # noqa: PERF102
            yield PandasLikeSeries.from_native(series, context=self)

    _iter_columns = iter_columns

    def iter_rows(
        self,
        *,
        named: bool,
        buffer_size: int,
    ) -> Iterator[tuple[Any, ...]] | Iterator[dict[str, Any]]:
        # The param ``buffer_size`` is only here for compatibility with the Polars API
        # and has no effect on the output.
        if not named:
            yield from self.native.itertuples(index=False, name=None)
        else:
            col_names = self.native.columns
            for row in self.native.itertuples(index=False):
                yield dict(zip(col_names, row))

    @property
    def schema(self) -> dict[str, DType]:
        native_dtypes = self.native.dtypes
        return {
            col: native_to_narwhals_dtype(
                native_dtypes[col], self._version, self._implementation
            )
            if native_dtypes[col] != "object"
            else object_native_to_narwhals_dtype(
                self.native[col], self._version, self._implementation
            )
            for col in self.native.columns
        }

    def collect_schema(self) -> dict[str, DType]:
        return self.schema

    # --- reshape ---
    def simple_select(self, *column_names: str) -> Self:
        return self._with_native(
            select_columns_by_name(
                self.native,
                list(column_names),
                self._backend_version,
                self._implementation,
            ),
            validate_column_names=False,
        )

    def select(self: PandasLikeDataFrame, *exprs: PandasLikeExpr) -> PandasLikeDataFrame:
        new_series = self._evaluate_into_exprs(*exprs)
        if not new_series:
            # return empty dataframe, like Polars does
            return self._with_native(self.native.__class__(), validate_column_names=False)
        new_series = align_series_full_broadcast(*new_series)
        namespace = self.__narwhals_namespace__()
        df = namespace._concat_horizontal([s.native for s in new_series])
        return self._with_native(df, validate_column_names=True)

    def drop_nulls(
        self: PandasLikeDataFrame, subset: Sequence[str] | None
    ) -> PandasLikeDataFrame:
        if subset is None:
            return self._with_native(
                self.native.dropna(axis=0), validate_column_names=False
            )
        plx = self.__narwhals_namespace__()
        return self.filter(~plx.any_horizontal(plx.col(*subset).is_null()))

    def estimated_size(self, unit: SizeUnit) -> int | float:
        sz = self.native.memory_usage(deep=True).sum()
        return scale_bytes(sz, unit=unit)

    def with_row_index(self, name: str) -> Self:
        frame = self.native
        namespace = self.__narwhals_namespace__()
        row_index = namespace._series.from_iterable(
            range(len(frame)), context=self, index=frame.index
        ).alias(name)
        return self._with_native(namespace._concat_horizontal([row_index.native, frame]))

    def row(self, index: int) -> tuple[Any, ...]:
        return tuple(x for x in self.native.iloc[index])

    def filter(
        self: PandasLikeDataFrame, predicate: PandasLikeExpr | list[bool]
    ) -> PandasLikeDataFrame:
        if isinstance(predicate, list):
            mask_native: pd.Series[Any] | list[bool] = predicate
        else:
            # `[0]` is safe as the predicate's expression only returns a single column
            mask = self._evaluate_into_exprs(predicate)[0]
            mask_native = self._extract_comparand(mask)
        return self._with_native(
            self.native.loc[mask_native], validate_column_names=False
        )

    def with_columns(
        self: PandasLikeDataFrame, *exprs: PandasLikeExpr
    ) -> PandasLikeDataFrame:
        columns = self._evaluate_into_exprs(*exprs)
        if not columns and len(self) == 0:
            return self
        name_columns: dict[str, PandasLikeSeries] = {s.name: s for s in columns}
        to_concat = []
        # Make sure to preserve column order
        for name in self.native.columns:
            if name in name_columns:
                series = self._extract_comparand(name_columns.pop(name))
            else:
                series = self.native[name]
            to_concat.append(series)
        to_concat.extend(self._extract_comparand(s) for s in name_columns.values())
        namespace = self.__narwhals_namespace__()
        df = namespace._concat_horizontal(to_concat)
        return self._with_native(df, validate_column_names=False)

    def rename(self, mapping: Mapping[str, str]) -> Self:
        return self._with_native(
            rename(
                self.native,
                columns=mapping,
                implementation=self._implementation,
                backend_version=self._backend_version,
            )
        )

    def drop(self, columns: Sequence[str], *, strict: bool) -> Self:
        to_drop = parse_columns_to_drop(
            compliant_frame=self, columns=columns, strict=strict
        )
        return self._with_native(
            self.native.drop(columns=to_drop), validate_column_names=False
        )

    # --- transform ---
    def sort(
        self,
        *by: str,
        descending: bool | Sequence[bool],
        nulls_last: bool,
    ) -> Self:
        df = self.native
        if isinstance(descending, bool):
            ascending: bool | list[bool] = not descending
        else:
            ascending = [not d for d in descending]
        na_position = "last" if nulls_last else "first"
        return self._with_native(
            df.sort_values(list(by), ascending=ascending, na_position=na_position),
            validate_column_names=False,
        )

    # --- convert ---
    def collect(
        self,
        backend: Implementation | None,
        **kwargs: Any,
    ) -> CompliantDataFrame[Any, Any, Any]:
        if backend is None:
            return PandasLikeDataFrame(
                self.native,
                implementation=self._implementation,
                backend_version=self._backend_version,
                version=self._version,
                validate_column_names=False,
            )

        if backend is Implementation.PANDAS:
            import pandas as pd  # ignore-banned-import

            return PandasLikeDataFrame(
                self.to_pandas(),
                implementation=Implementation.PANDAS,
                backend_version=parse_version(pd),
                version=self._version,
                validate_column_names=False,
            )

        if backend is Implementation.PYARROW:
            import pyarrow as pa  # ignore-banned-import

            from narwhals._arrow.dataframe import ArrowDataFrame

            return ArrowDataFrame(
                native_dataframe=self.to_arrow(),
                backend_version=parse_version(pa),
                version=self._version,
                validate_column_names=False,
            )

        if backend is Implementation.POLARS:
            import polars as pl  # ignore-banned-import

            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame(
                df=self.to_polars(),
                backend_version=parse_version(pl),
                version=self._version,
            )

        msg = f"Unsupported `backend` value: {backend}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    # --- actions ---
    def group_by(
        self, keys: Sequence[str] | Sequence[PandasLikeExpr], *, drop_null_keys: bool
    ) -> PandasLikeGroupBy:
        from narwhals._pandas_like.group_by import PandasLikeGroupBy

        return PandasLikeGroupBy(self, keys, drop_null_keys=drop_null_keys)

    def join(
        self,
        other: Self,
        *,
        how: JoinStrategy,
        left_on: Sequence[str] | None,
        right_on: Sequence[str] | None,
        suffix: str,
    ) -> Self:
        if how == "cross":
            if (
                self._implementation is Implementation.MODIN
                or self._implementation is Implementation.CUDF
            ) or (
                self._implementation is Implementation.PANDAS
                and self._backend_version < (1, 4)
            ):
                key_token = generate_temporary_column_name(
                    n_bytes=8, columns=[*self.columns, *other.columns]
                )

                return self._with_native(
                    self.native.assign(**{key_token: 0})
                    .merge(
                        other.native.assign(**{key_token: 0}),
                        how="inner",
                        left_on=key_token,
                        right_on=key_token,
                        suffixes=("", suffix),
                    )
                    .drop(columns=key_token)
                )
            else:
                return self._with_native(
                    self.native.merge(other.native, how="cross", suffixes=("", suffix))
                )

        if how == "anti":
            if self._implementation is Implementation.CUDF:
                return self._with_native(
                    self.native.merge(
                        other.native, how="leftanti", left_on=left_on, right_on=right_on
                    )
                )
            else:
                indicator_token = generate_temporary_column_name(
                    n_bytes=8, columns=[*self.columns, *other.columns]
                )
                if right_on is None:  # pragma: no cover
                    msg = "`right_on` cannot be `None` in anti-join"
                    raise TypeError(msg)

                # rename to avoid creating extra columns in join
                other_native = rename(
                    select_columns_by_name(
                        other.native,
                        list(right_on),
                        self._backend_version,
                        self._implementation,
                    ),
                    columns=dict(zip(right_on, left_on)),  # type: ignore[arg-type]
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                ).drop_duplicates()
                return self._with_native(
                    self.native.merge(
                        other_native,
                        how="outer",
                        indicator=indicator_token,
                        left_on=left_on,
                        right_on=left_on,
                    )
                    .loc[lambda t: t[indicator_token] == "left_only"]
                    .drop(columns=indicator_token)
                )

        if how == "semi":
            if right_on is None:  # pragma: no cover
                msg = "`right_on` cannot be `None` in semi-join"
                raise TypeError(msg)
            # rename to avoid creating extra columns in join
            other_native = (
                rename(
                    select_columns_by_name(
                        other.native,
                        list(right_on),
                        self._backend_version,
                        self._implementation,
                    ),
                    columns=dict(zip(right_on, left_on)),  # type: ignore[arg-type]
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                ).drop_duplicates()  # avoids potential rows duplication from inner join
            )
            return self._with_native(
                self.native.merge(
                    other_native, how="inner", left_on=left_on, right_on=left_on
                )
            )

        if how == "left":
            result_native = self.native.merge(
                other.native,
                how="left",
                left_on=left_on,
                right_on=right_on,
                suffixes=("", suffix),
            )
            extra = []
            for left_key, right_key in zip(left_on, right_on):  # type: ignore[arg-type]
                if right_key != left_key and right_key not in self.columns:
                    extra.append(right_key)
                elif right_key != left_key:
                    extra.append(f"{right_key}{suffix}")
            return self._with_native(result_native.drop(columns=extra))

        if how == "full":
            # Pandas coalesces keys in full joins unless there's no collision

            # help mypy
            assert left_on is not None  # noqa: S101
            assert right_on is not None  # noqa: S101

            right_on_mapper = _remap_full_join_keys(left_on, right_on, suffix)
            other_native = other.native.rename(columns=right_on_mapper)
            check_column_names_are_unique(other_native.columns)
            right_on = list(right_on_mapper.values())  # we now have the suffixed keys
            return self._with_native(
                self.native.merge(
                    other_native,
                    left_on=left_on,
                    right_on=right_on,
                    how="outer",
                    suffixes=("", suffix),
                ),
            )

        return self._with_native(
            self.native.merge(
                other.native,
                left_on=left_on,
                right_on=right_on,
                how=how,
                suffixes=("", suffix),
            )
        )

    def join_asof(
        self,
        other: Self,
        *,
        left_on: str | None,
        right_on: str | None,
        by_left: Sequence[str] | None,
        by_right: Sequence[str] | None,
        strategy: AsofJoinStrategy,
        suffix: str,
    ) -> Self:
        plx = self.__native_namespace__()
        return self._with_native(
            plx.merge_asof(
                self.native,
                other.native,
                left_on=left_on,
                right_on=right_on,
                left_by=by_left,
                right_by=by_right,
                direction=strategy,
                suffixes=("", suffix),
            ),
        )

    # --- partial reduction ---

    def head(self, n: int) -> Self:
        return self._with_native(self.native.head(n), validate_column_names=False)

    def tail(self, n: int) -> Self:
        return self._with_native(self.native.tail(n), validate_column_names=False)

    def unique(
        self,
        subset: Sequence[str] | None,
        *,
        keep: UniqueKeepStrategy,
        maintain_order: bool | None = None,
    ) -> Self:
        # The param `maintain_order` is only here for compatibility with the Polars API
        # and has no effect on the output.
        mapped_keep = {"none": False, "any": "first"}.get(keep, keep)
        check_column_exists(self.columns, subset)
        return self._with_native(
            self.native.drop_duplicates(subset=subset, keep=mapped_keep),
            validate_column_names=False,
        )

    # --- lazy-only ---
    def lazy(
        self, *, backend: Implementation | None = None
    ) -> CompliantLazyFrame[Any, Any]:
        from narwhals.utils import parse_version

        pandas_df = self.to_pandas()
        if backend is None:
            return self
        elif backend is Implementation.DUCKDB:
            import duckdb  # ignore-banned-import

            from narwhals._duckdb.dataframe import DuckDBLazyFrame

            return DuckDBLazyFrame(
                df=duckdb.table("pandas_df"),
                backend_version=parse_version(duckdb),
                version=self._version,
            )
        elif backend is Implementation.POLARS:
            import polars as pl  # ignore-banned-import

            from narwhals._polars.dataframe import PolarsLazyFrame

            return PolarsLazyFrame(
                df=pl.from_pandas(pandas_df).lazy(),
                backend_version=parse_version(pl),
                version=self._version,
            )
        elif backend is Implementation.DASK:
            import dask  # ignore-banned-import
            import dask.dataframe as dd  # ignore-banned-import

            from narwhals._dask.dataframe import DaskLazyFrame

            return DaskLazyFrame(
                native_dataframe=dd.from_pandas(pandas_df),
                backend_version=parse_version(dask),
                version=self._version,
            )
        raise AssertionError  # pragma: no cover

    @property
    def shape(self) -> tuple[int, int]:
        return self.native.shape

    def to_dict(self, *, as_series: bool) -> dict[str, Any]:
        if as_series:
            return {
                col: PandasLikeSeries.from_native(self.native[col], context=self)
                for col in self.columns
            }
        return self.native.to_dict(orient="list")

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _2DArray:
        native_dtypes = self.native.dtypes

        if copy is None:
            # pandas default differs from Polars, but cuDF default is True
            copy = self._implementation is Implementation.CUDF

        if native_dtypes.isin(CLASSICAL_NUMPY_DTYPES).all():
            # Fast path, no conversions necessary.
            if dtype is not None:
                return self.native.to_numpy(dtype=dtype, copy=copy)
            return self.native.to_numpy(copy=copy)

        dtype_datetime = self._version.dtypes.Datetime
        to_convert = [
            key
            for key, val in self.schema.items()
            if isinstance(val, dtype_datetime) and val.time_zone is not None
        ]
        if to_convert:
            df = self.with_columns(
                self.__narwhals_namespace__()
                .col(*to_convert)
                .dt.convert_time_zone("UTC")
                .dt.replace_time_zone(None)
            ).native
        else:
            df = self.native

        if dtype is not None:
            return df.to_numpy(dtype=dtype, copy=copy)

        # pandas return `object` dtype for nullable dtypes if dtype=None,
        # so we cast each Series to numpy and let numpy find a common dtype.
        # If there aren't any dtypes where `to_numpy()` is "broken" (i.e. it
        # returns Object) then we just call `to_numpy()` on the DataFrame.
        for col_dtype in native_dtypes:
            if str(col_dtype) in PANDAS_TO_NUMPY_DTYPE_MISSING:
                import numpy as np

                arr: Any = np.hstack(
                    [
                        self.get_column(col).to_numpy(copy=copy, dtype=None)[:, None]
                        for col in self.columns
                    ]
                )
                return arr
        return df.to_numpy(copy=copy)

    def to_pandas(self) -> pd.DataFrame:
        if self._implementation is Implementation.PANDAS:
            return self.native
        elif self._implementation is Implementation.CUDF:
            return self.native.to_pandas()
        elif self._implementation is Implementation.MODIN:
            return self.native._to_pandas()
        msg = f"Unknown implementation: {self._implementation}"  # pragma: no cover
        raise AssertionError(msg)

    def to_polars(self) -> pl.DataFrame:
        import polars as pl  # ignore-banned-import

        return pl.from_pandas(self.to_pandas())

    def write_parquet(self, file: str | Path | BytesIO) -> None:
        self.native.to_parquet(file)

    @overload
    def write_csv(self, file: None) -> str: ...

    @overload
    def write_csv(self, file: str | Path | BytesIO) -> None: ...

    def write_csv(self, file: str | Path | BytesIO | None) -> str | None:
        return self.native.to_csv(file, index=False)

    # --- descriptive ---
    def is_unique(self) -> PandasLikeSeries:
        return PandasLikeSeries.from_native(
            ~self.native.duplicated(keep=False), context=self
        )

    def item(self, row: int | None, column: int | str | None) -> Any:
        if row is None and column is None:
            if self.shape != (1, 1):
                msg = (
                    "can only call `.item()` if the dataframe is of shape (1, 1),"
                    " or if explicit row/col values are provided;"
                    f" frame has shape {self.shape!r}"
                )
                raise ValueError(msg)
            return self.native.iloc[0, 0]

        elif row is None or column is None:
            msg = "cannot call `.item()` with only one of `row` or `column`"
            raise ValueError(msg)

        _col = self.columns.index(column) if isinstance(column, str) else column
        return self.native.iloc[row, _col]

    def clone(self) -> Self:
        return self._with_native(self.native.copy(), validate_column_names=False)

    def gather_every(self, n: int, offset: int) -> Self:
        return self._with_native(self.native.iloc[offset::n], validate_column_names=False)

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
        if self._implementation is Implementation.PANDAS and (
            self._backend_version < (1, 1)
        ):  # pragma: no cover
            msg = "pivot is only supported for pandas>=1.1"
            raise NotImplementedError(msg)
        if self._implementation is Implementation.MODIN:
            msg = "pivot is not supported for Modin backend due to https://github.com/modin-project/modin/issues/7409."
            raise NotImplementedError(msg)
        from itertools import product

        frame = self.native

        if index is None:
            index = [c for c in self.columns if c not in {*on, *values}]  # type: ignore[misc]

        if values is None:
            values = [c for c in self.columns if c not in {*on, *index}]

        if aggregate_function is None:
            result = frame.pivot(columns=on, index=index, values=values)
        elif aggregate_function == "len":
            result = (
                frame.groupby([*on, *index])
                .agg(dict.fromkeys(values, "size"))
                .reset_index()
                .pivot(columns=on, index=index, values=values)
            )
        else:
            result = pivot_table(
                df=self,
                values=values,
                index=index,
                columns=on,
                aggregate_function=aggregate_function,
            )

        # Put columns in the right order
        if sort_columns and self._implementation is Implementation.CUDF:
            uniques = {
                col: sorted(self.native[col].unique().to_arrow().to_pylist())
                for col in on
            }
        elif sort_columns:
            uniques = {col: sorted(self.native[col].unique().tolist()) for col in on}
        elif self._implementation is Implementation.CUDF:
            uniques = {
                col: self.native[col].unique().to_arrow().to_pylist() for col in on
            }
        else:
            uniques = {col: self.native[col].unique().tolist() for col in on}
        ordered_cols = list(product(values, *uniques.values()))
        result = result.loc[:, ordered_cols]
        columns = result.columns.tolist()

        n_on = len(on)
        if n_on == 1:
            new_columns = [
                separator.join(col).strip() if len(values) > 1 else col[-1]
                for col in columns
            ]
        else:
            new_columns = [
                separator.join([col[0], '{"' + '","'.join(col[-n_on:]) + '"}'])
                if len(values) > 1
                else '{"' + '","'.join(col[-n_on:]) + '"}'
                for col in columns
            ]
        result.columns = new_columns
        result.columns.names = [""]  # type: ignore[attr-defined]
        return self._with_native(result.reset_index())

    def to_arrow(self) -> Any:
        if self._implementation is Implementation.CUDF:
            return self.native.to_arrow(preserve_index=False)

        import pyarrow as pa  # ignore-banned-import()

        return pa.Table.from_pandas(self.native)

    def sample(
        self,
        n: int | None,
        *,
        fraction: float | None,
        with_replacement: bool,
        seed: int | None,
    ) -> Self:
        return self._with_native(
            self.native.sample(
                n=n, frac=fraction, replace=with_replacement, random_state=seed
            ),
            validate_column_names=False,
        )

    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self:
        return self._with_native(
            self.native.melt(
                id_vars=index,
                value_vars=on,
                var_name=variable_name,
                value_name=value_name,
            )
        )

    def explode(self, columns: Sequence[str]) -> Self:
        dtypes = self._version.dtypes

        schema = self.collect_schema()
        for col_to_explode in columns:
            dtype = schema[col_to_explode]

            if dtype != dtypes.List:
                msg = (
                    f"`explode` operation not supported for dtype `{dtype}`, "
                    "expected List type"
                )
                raise InvalidOperationError(msg)

        if len(columns) == 1:
            return self._with_native(
                self.native.explode(columns[0]), validate_column_names=False
            )
        else:
            native_frame = self.native
            anchor_series = native_frame[columns[0]].list.len()

            if not all(
                (native_frame[col_name].list.len() == anchor_series).all()
                for col_name in columns[1:]
            ):
                msg = "exploded columns must have matching element counts"
                raise ShapeError(msg)

            original_columns = self.columns
            other_columns = [c for c in original_columns if c not in columns]

            exploded_frame = native_frame[[*other_columns, columns[0]]].explode(
                columns[0]
            )
            exploded_series = [
                native_frame[col_name].explode().to_frame() for col_name in columns[1:]
            ]

            plx = self.__native_namespace__()
            return self._with_native(
                plx.concat([exploded_frame, *exploded_series], axis=1)[original_columns],
                validate_column_names=False,
            )
