from __future__ import annotations

import collections
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import Sequence
from typing import overload

from narwhals._expression_parsing import evaluate_into_exprs
from narwhals._pandas_like.expr import PandasLikeExpr
from narwhals._pandas_like.utils import broadcast_series
from narwhals._pandas_like.utils import create_native_series
from narwhals._pandas_like.utils import horizontal_concat
from narwhals._pandas_like.utils import translate_dtype
from narwhals._pandas_like.utils import validate_dataframe_comparand
from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_pandas
from narwhals.dependencies import is_numpy_array
from narwhals.utils import Implementation
from narwhals.utils import flatten
from narwhals.utils import generate_unique_token
from narwhals.utils import parse_columns_to_drop

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.group_by import PandasLikeGroupBy
    from narwhals._pandas_like.namespace import PandasLikeNamespace
    from narwhals._pandas_like.series import PandasLikeSeries
    from narwhals._pandas_like.typing import IntoPandasLikeExpr
    from narwhals.dtypes import DType


class PandasLikeDataFrame:
    # --- not in the spec ---
    def __init__(
        self,
        native_dataframe: Any,
        *,
        implementation: Implementation,
        backend_version: tuple[int, ...],
    ) -> None:
        self._validate_columns(native_dataframe.columns)
        self._native_frame = native_dataframe
        self._implementation = implementation
        self._backend_version = backend_version

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PandasLikeNamespace:
        from narwhals._pandas_like.namespace import PandasLikeNamespace

        return PandasLikeNamespace(self._implementation, self._backend_version)

    def __native_namespace__(self) -> Any:
        if self._implementation is Implementation.PANDAS:
            return get_pandas()
        if self._implementation is Implementation.MODIN:  # pragma: no cover
            return get_modin()
        if self._implementation is Implementation.CUDF:  # pragma: no cover
            return get_cudf()
        msg = f"Expected pandas/modin/cudf, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def __len__(self) -> int:
        return len(self._native_frame)

    def _validate_columns(self, columns: Sequence[str]) -> None:
        if len(columns) != len(set(columns)):
            counter = collections.Counter(columns)
            for col, count in counter.items():
                if count > 1:
                    msg = f"Expected unique column names, got {col!r} {count} time(s)"
                    raise ValueError(msg)
            msg = "Please report a bug"  # pragma: no cover
            raise AssertionError(msg)

    def _from_native_frame(self, df: Any) -> Self:
        return self.__class__(
            df,
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    def get_column(self, name: str) -> PandasLikeSeries:
        from narwhals._pandas_like.series import PandasLikeSeries

        return PandasLikeSeries(
            self._native_frame.loc[:, name],
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    @overload
    def __getitem__(self, item: tuple[Sequence[int], str | int]) -> PandasLikeSeries: ...  # type: ignore[overload-overlap]

    @overload
    def __getitem__(self, item: Sequence[int]) -> PandasLikeDataFrame: ...

    @overload
    def __getitem__(self, item: str) -> PandasLikeSeries: ...

    @overload
    def __getitem__(self, item: slice) -> PandasLikeDataFrame: ...

    def __getitem__(
        self, item: str | slice | Sequence[int] | tuple[Sequence[int], str | int]
    ) -> PandasLikeSeries | PandasLikeDataFrame:
        if isinstance(item, str):
            from narwhals._pandas_like.series import PandasLikeSeries

            return PandasLikeSeries(
                self._native_frame.loc[:, item],
                implementation=self._implementation,
                backend_version=self._backend_version,
            )

        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[1], (tuple, list))
        ):
            if all(isinstance(x, int) for x in item[1]):
                return self._from_native_frame(self._native_frame.iloc[item])
            if all(isinstance(x, str) for x in item[1]):
                item = (
                    item[0],
                    self._native_frame.columns.get_indexer(item[1]),
                )
                return self._from_native_frame(self._native_frame.iloc[item])
            msg = (
                f"Expected sequence str or int, got: {type(item[1])}"  # pragma: no cover
            )
            raise TypeError(msg)  # pragma: no cover

        elif isinstance(item, tuple) and len(item) == 2:
            from narwhals._pandas_like.series import PandasLikeSeries

            if isinstance(item[1], str):
                item = (item[0], self._native_frame.columns.get_loc(item[1]))
                native_series = self._native_frame.iloc[item]
            elif isinstance(item[1], int):
                native_series = self._native_frame.iloc[item]
            else:  # pragma: no cover
                msg = f"Expected str or int, got: {type(item[1])}"
                raise TypeError(msg)

            return PandasLikeSeries(
                native_series,
                implementation=self._implementation,
                backend_version=self._backend_version,
            )

        elif isinstance(item, (slice, Sequence)) or (
            is_numpy_array(item) and item.ndim == 1
        ):
            return self._from_native_frame(self._native_frame.iloc[item])

        else:  # pragma: no cover
            msg = f"Expected str or slice, got: {type(item)}"
            raise TypeError(msg)

    # --- properties ---
    @property
    def columns(self) -> list[str]:
        return self._native_frame.columns.tolist()  # type: ignore[no-any-return]

    def rows(
        self, *, named: bool = False
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        if not named:
            return list(self._native_frame.itertuples(index=False, name=None))

        return self._native_frame.to_dict(orient="records")  # type: ignore[no-any-return]

    def iter_rows(
        self,
        *,
        named: bool = False,
        buffer_size: int = 512,
    ) -> Iterator[list[tuple[Any, ...]]] | Iterator[list[dict[str, Any]]]:
        """
        NOTE:
            The param ``buffer_size`` is only here for compatibility with the polars API
            and has no effect on the output.
        """
        if not named:
            yield from self._native_frame.itertuples(index=False, name=None)
        else:
            col_names = self._native_frame.columns
            yield from (
                dict(zip(col_names, row))
                for row in self._native_frame.itertuples(index=False)
            )  # type: ignore[misc]

    @property
    def schema(self) -> dict[str, DType]:
        return {
            col: translate_dtype(self._native_frame.loc[:, col])
            for col in self._native_frame.columns
        }

    def collect_schema(self) -> dict[str, DType]:
        return self.schema

    # --- reshape ---
    def select(
        self,
        *exprs: IntoPandasLikeExpr,
        **named_exprs: IntoPandasLikeExpr,
    ) -> Self:
        if exprs and all(isinstance(x, str) for x in exprs) and not named_exprs:
            # This is a simple slice => fastpath!
            return self._from_native_frame(self._native_frame.loc[:, list(exprs)])
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        if not new_series:
            # return empty dataframe, like Polars does
            return self._from_native_frame(self._native_frame.__class__())
        new_series = broadcast_series(new_series)
        df = horizontal_concat(
            new_series,
            implementation=self._implementation,
            backend_version=self._backend_version,
        )
        return self._from_native_frame(df)

    def drop_nulls(self, subset: str | list[str] | None) -> Self:
        if subset is None:
            return self._from_native_frame(self._native_frame.dropna(axis=0))
        subset = [subset] if isinstance(subset, str) else subset
        plx = self.__narwhals_namespace__()
        return self.filter(~plx.any_horizontal(plx.col(*subset).is_null()))

    def with_row_index(self, name: str) -> Self:
        row_index = create_native_series(
            range(len(self._native_frame)),
            index=self._native_frame.index,
            implementation=self._implementation,
            backend_version=self._backend_version,
        ).alias(name)
        return self._from_native_frame(
            horizontal_concat(
                [row_index._native_series, self._native_frame],
                implementation=self._implementation,
                backend_version=self._backend_version,
            )
        )

    def filter(
        self,
        *predicates: IntoPandasLikeExpr,
    ) -> Self:
        plx = self.__narwhals_namespace__()
        expr = plx.all_horizontal(*predicates)
        # Safety: all_horizontal's expression only returns a single column.
        mask = expr._call(self)[0]
        _mask = validate_dataframe_comparand(self._native_frame.index, mask)
        return self._from_native_frame(self._native_frame.loc[_mask])

    def with_columns(
        self,
        *exprs: IntoPandasLikeExpr,
        **named_exprs: IntoPandasLikeExpr,
    ) -> Self:
        index = self._native_frame.index
        new_columns = evaluate_into_exprs(self, *exprs, **named_exprs)

        if not new_columns and len(self) == 0:
            return self

        # If the inputs are all Expressions which return full columns
        # (as opposed to scalars), we can use a fast path (concat, instead of assign).
        # We can't use the fastpath if any input is not an expression (e.g.
        # if it's a Series) because then we might be changing its flags.
        # See `test_memmap` for an example of where this is necessary.
        fast_path = (
            all(len(s) > 1 for s in new_columns)
            and all(isinstance(x, PandasLikeExpr) for x in exprs)
            and all(isinstance(x, PandasLikeExpr) for (_, x) in named_exprs.items())
        )

        if fast_path:
            new_column_name_to_new_column_map = {s.name: s for s in new_columns}
            to_concat = []
            # Make sure to preserve column order
            for name in self._native_frame.columns:
                if name in new_column_name_to_new_column_map:
                    to_concat.append(
                        validate_dataframe_comparand(
                            index, new_column_name_to_new_column_map.pop(name)
                        )
                    )
                else:
                    to_concat.append(self._native_frame.loc[:, name])
            to_concat.extend(
                validate_dataframe_comparand(index, new_column_name_to_new_column_map[s])
                for s in new_column_name_to_new_column_map
            )

            df = horizontal_concat(
                to_concat,
                implementation=self._implementation,
                backend_version=self._backend_version,
            )
        else:
            df = self._native_frame.copy(deep=False)
            for s in new_columns:
                df[s.name] = validate_dataframe_comparand(index, s)
        return self._from_native_frame(df)

    def rename(self, mapping: dict[str, str]) -> Self:
        return self._from_native_frame(self._native_frame.rename(columns=mapping))

    def drop(self: Self, columns: list[str], strict: bool) -> Self:  # noqa: FBT001
        to_drop = parse_columns_to_drop(
            compliant_frame=self, columns=columns, strict=strict
        )
        return self._from_native_frame(self._native_frame.drop(columns=to_drop))

    # --- transform ---
    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        flat_keys = flatten([*flatten([by]), *more_by])
        df = self._native_frame
        if isinstance(descending, bool):
            ascending: bool | list[bool] = not descending
        else:
            ascending = [not d for d in descending]
        return self._from_native_frame(df.sort_values(flat_keys, ascending=ascending))

    # --- convert ---
    def collect(self) -> PandasLikeDataFrame:
        return PandasLikeDataFrame(
            self._native_frame,
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    # --- actions ---
    def group_by(self, *keys: str) -> PandasLikeGroupBy:
        from narwhals._pandas_like.group_by import PandasLikeGroupBy

        return PandasLikeGroupBy(
            self,
            list(keys),
        )

    def join(
        self,
        other: Self,
        *,
        how: Literal["left", "inner", "outer", "cross", "anti", "semi"] = "inner",
        left_on: str | list[str] | None,
        right_on: str | list[str] | None,
    ) -> Self:
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        if how == "cross":
            if (
                self._implementation is Implementation.MODIN
                or self._implementation is Implementation.CUDF
            ) or (
                self._implementation is Implementation.PANDAS
                and self._backend_version < (1, 4)
            ):
                key_token = generate_unique_token(
                    n_bytes=8, columns=[*self.columns, *other.columns]
                )

                return self._from_native_frame(
                    self._native_frame.assign(**{key_token: 0})
                    .merge(
                        other._native_frame.assign(**{key_token: 0}),
                        how="inner",
                        left_on=key_token,
                        right_on=key_token,
                        suffixes=("", "_right"),
                    )
                    .drop(columns=key_token),
                )
            else:
                return self._from_native_frame(
                    self._native_frame.merge(
                        other._native_frame,
                        how="cross",
                        suffixes=("", "_right"),
                    ),
                )

        if how == "anti":
            indicator_token = generate_unique_token(
                n_bytes=8, columns=[*self.columns, *other.columns]
            )

            other_native = (
                other._native_frame.loc[:, right_on]
                .rename(  # rename to avoid creating extra columns in join
                    columns=dict(zip(right_on, left_on))  # type: ignore[arg-type]
                )
                .drop_duplicates()
            )
            return self._from_native_frame(
                self._native_frame.merge(
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
            other_native = (
                other._native_frame.loc[:, right_on]
                .rename(  # rename to avoid creating extra columns in join
                    columns=dict(zip(right_on, left_on))  # type: ignore[arg-type]
                )
                .drop_duplicates()  # avoids potential rows duplication from inner join
            )
            return self._from_native_frame(
                self._native_frame.merge(
                    other_native,
                    how="inner",
                    left_on=left_on,
                    right_on=left_on,
                )
            )

        if how == "left":
            other_native = other._native_frame
            result_native = self._native_frame.merge(
                other_native,
                how="left",
                left_on=left_on,
                right_on=right_on,
                suffixes=("", "_right"),
            )
            extra = []
            for left_key, right_key in zip(left_on, right_on):  # type: ignore[arg-type]
                if right_key != left_key and right_key not in self.columns:
                    extra.append(right_key)
                elif right_key != left_key:
                    extra.append(f"{right_key}_right")
            return self._from_native_frame(result_native.drop(columns=extra))

        return self._from_native_frame(
            self._native_frame.merge(
                other._native_frame,
                left_on=left_on,
                right_on=right_on,
                how=how,
                suffixes=("", "_right"),
            ),
        )

    # --- partial reduction ---

    def head(self, n: int) -> Self:
        return self._from_native_frame(self._native_frame.head(n))

    def tail(self, n: int) -> Self:
        return self._from_native_frame(self._native_frame.tail(n))

    def unique(
        self: Self,
        subset: str | list[str] | None,
        *,
        keep: Literal["any", "first", "last", "none"] = "any",
        maintain_order: bool = False,
    ) -> Self:
        """
        NOTE:
            The param `maintain_order` is only here for compatibility with the polars API
            and has no effect on the output.
        """
        mapped_keep = {"none": False, "any": "first"}.get(keep, keep)
        subset = flatten(subset) if subset else None
        return self._from_native_frame(
            self._native_frame.drop_duplicates(subset=subset, keep=mapped_keep)
        )

    # --- lazy-only ---
    def lazy(self) -> Self:
        return self

    @property
    def shape(self) -> tuple[int, int]:
        return self._native_frame.shape  # type: ignore[no-any-return]

    def to_dict(self, *, as_series: bool = False) -> dict[str, Any]:
        from narwhals._pandas_like.series import PandasLikeSeries

        if as_series:
            # TODO(Unassigned): should this return narwhals series?
            return {
                col: PandasLikeSeries(
                    self._native_frame.loc[:, col],
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                )
                for col in self.columns
            }
        return self._native_frame.to_dict(orient="list")  # type: ignore[no-any-return]

    def to_numpy(self) -> Any:
        from narwhals._pandas_like.series import PANDAS_TO_NUMPY_DTYPE_MISSING

        # pandas return `object` dtype for nullable dtypes, so we cast each
        # Series to numpy and let numpy find a common dtype.
        # If there aren't any dtypes where `to_numpy()` is "broken" (i.e. it
        # returns Object) then we just call `to_numpy()` on the DataFrame.
        for dtype in self._native_frame.dtypes:
            if str(dtype) in PANDAS_TO_NUMPY_DTYPE_MISSING:
                import numpy as np  # ignore-banned-import

                return np.hstack([self[col].to_numpy()[:, None] for col in self.columns])
        return self._native_frame.to_numpy()

    def to_pandas(self) -> Any:
        if self._implementation is Implementation.PANDAS:
            return self._native_frame
        if self._implementation is Implementation.MODIN:  # pragma: no cover
            return self._native_frame._to_pandas()
        return self._native_frame.to_pandas()  # pragma: no cover

    def write_parquet(self, file: Any) -> Any:
        self._native_frame.to_parquet(file)

    # --- descriptive ---
    def is_duplicated(self: Self) -> PandasLikeSeries:
        from narwhals._pandas_like.series import PandasLikeSeries

        return PandasLikeSeries(
            self._native_frame.duplicated(keep=False),
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    def is_empty(self: Self) -> bool:
        return self._native_frame.empty  # type: ignore[no-any-return]

    def is_unique(self: Self) -> PandasLikeSeries:
        from narwhals._pandas_like.series import PandasLikeSeries

        return PandasLikeSeries(
            ~self._native_frame.duplicated(keep=False),
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    def null_count(self: Self) -> PandasLikeDataFrame:
        return PandasLikeDataFrame(
            self._native_frame.isna().sum(axis=0).to_frame().transpose(),
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    def item(self: Self, row: int | None = None, column: int | str | None = None) -> Any:
        if row is None and column is None:
            if self.shape != (1, 1):
                msg = (
                    "can only call `.item()` if the dataframe is of shape (1, 1),"
                    " or if explicit row/col values are provided;"
                    f" frame has shape {self.shape!r}"
                )
                raise ValueError(msg)
            return self._native_frame.iloc[0, 0]

        elif row is None or column is None:
            msg = "cannot call `.item()` with only one of `row` or `column`"
            raise ValueError(msg)

        _col = self.columns.index(column) if isinstance(column, str) else column
        return self._native_frame.iloc[row, _col]

    def clone(self: Self) -> Self:
        return self._from_native_frame(self._native_frame.copy())

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        return self._from_native_frame(self._native_frame.iloc[offset::n])

    def to_arrow(self: Self) -> Any:
        if self._implementation is Implementation.CUDF:  # pragma: no cover
            msg = "`to_arrow` is not implemented for CuDF backend."
            raise NotImplementedError(msg)

        import pyarrow as pa  # ignore-banned-import()

        return pa.Table.from_pandas(self._native_frame)
