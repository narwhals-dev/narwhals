from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import Sequence
from typing import overload

from narwhals._arrow.utils import broadcast_series
from narwhals._arrow.utils import translate_dtype
from narwhals._arrow.utils import validate_dataframe_comparand
from narwhals._expression_parsing import evaluate_into_exprs
from narwhals.dependencies import get_numpy
from narwhals.dependencies import get_pyarrow
from narwhals.dependencies import get_pyarrow_compute
from narwhals.dependencies import get_pyarrow_parquet
from narwhals.utils import Implementation
from narwhals.utils import flatten
from narwhals.utils import generate_unique_token

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.group_by import ArrowGroupBy
    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals._arrow.series import ArrowSeries
    from narwhals._arrow.typing import IntoArrowExpr
    from narwhals.dtypes import DType


class ArrowDataFrame:
    # --- not in the spec ---
    def __init__(
        self, native_dataframe: Any, *, backend_version: tuple[int, ...]
    ) -> None:
        self._native_dataframe = native_dataframe
        self._implementation = Implementation.PYARROW
        self._backend_version = backend_version

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._arrow.namespace import ArrowNamespace

        return ArrowNamespace(backend_version=self._backend_version)

    def __native_namespace__(self) -> Any:
        return get_pyarrow()

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def _from_native_dataframe(self, df: Any) -> Self:
        return self.__class__(df, backend_version=self._backend_version)

    @property
    def shape(self) -> tuple[int, int]:
        return self._native_dataframe.shape  # type: ignore[no-any-return]

    def __len__(self) -> int:
        return len(self._native_dataframe)

    def rows(
        self, *, named: bool = False
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        if not named:
            msg = "Unnamed rows are not yet supported on PyArrow tables"
            raise NotImplementedError(msg)
        return self._native_dataframe.to_pylist()  # type: ignore[no-any-return]

    def iter_rows(
        self,
        *,
        named: bool = False,
        buffer_size: int = 512,
    ) -> Iterator[tuple[Any, ...]] | Iterator[dict[str, Any]]:
        df = self._native_dataframe
        num_rows = df.num_rows

        if not named:
            for i in range(0, num_rows, buffer_size):
                rows = df[i : i + buffer_size].to_pydict().values()
                yield from zip(*rows)
        else:
            for i in range(0, num_rows, buffer_size):
                yield from df[i : i + buffer_size].to_pylist()

    def get_column(self, name: str) -> ArrowSeries:
        from narwhals._arrow.series import ArrowSeries

        if not isinstance(name, str):
            msg = f"Expected str, got: {type(name)}"
            raise TypeError(msg)

        return ArrowSeries(
            self._native_dataframe[name],
            name=name,
            backend_version=self._backend_version,
        )

    @overload
    def __getitem__(self, item: tuple[Sequence[int], str | int]) -> ArrowSeries: ...  # type: ignore[overload-overlap]

    @overload
    def __getitem__(self, item: Sequence[int]) -> ArrowDataFrame: ...

    @overload
    def __getitem__(self, item: str) -> ArrowSeries: ...

    @overload
    def __getitem__(self, item: slice) -> ArrowDataFrame: ...

    def __getitem__(
        self, item: str | slice | Sequence[int] | tuple[Sequence[int], str | int]
    ) -> ArrowSeries | ArrowDataFrame:
        if isinstance(item, str):
            from narwhals._arrow.series import ArrowSeries

            return ArrowSeries(
                self._native_dataframe[item],
                name=item,
                backend_version=self._backend_version,
            )
        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[1], (list, tuple))
        ):
            return self._from_native_dataframe(
                self._native_dataframe.take(item[0]).select(item[1])
            )

        elif isinstance(item, tuple) and len(item) == 2:
            from narwhals._arrow.series import ArrowSeries

            # PyArrow columns are always strings
            col_name = item[1] if isinstance(item[1], str) else self.columns[item[1]]
            return ArrowSeries(
                self._native_dataframe[col_name].take(item[0]),
                name=col_name,
                backend_version=self._backend_version,
            )

        elif isinstance(item, slice):
            if item.step is not None and item.step != 1:
                msg = "Slicing with step is not supported on PyArrow tables"
                raise NotImplementedError(msg)
            start = item.start or 0
            stop = item.stop or len(self._native_dataframe)
            return self._from_native_dataframe(
                self._native_dataframe.slice(item.start, stop - start),
            )

        elif isinstance(item, Sequence) or (
            (np := get_numpy()) is not None
            and isinstance(item, np.ndarray)
            and item.ndim == 1
        ):
            return self._from_native_dataframe(self._native_dataframe.take(item))

        else:  # pragma: no cover
            msg = f"Expected str or slice, got: {type(item)}"
            raise TypeError(msg)

    @property
    def schema(self) -> dict[str, DType]:
        schema = self._native_dataframe.schema
        return {
            name: translate_dtype(dtype)
            for name, dtype in zip(schema.names, schema.types)
        }

    def collect_schema(self) -> dict[str, DType]:
        return self.schema

    @property
    def columns(self) -> list[str]:
        return self._native_dataframe.schema.names  # type: ignore[no-any-return]

    def select(
        self,
        *exprs: IntoArrowExpr,
        **named_exprs: IntoArrowExpr,
    ) -> Self:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        if not new_series:
            # return empty dataframe, like Polars does
            return self._from_native_dataframe(
                self._native_dataframe.__class__.from_arrays([])
            )
        names = [s.name for s in new_series]
        pa = get_pyarrow()
        df = pa.Table.from_arrays(
            broadcast_series(new_series),
            names=names,
        )
        return self._from_native_dataframe(df)

    def with_columns(
        self,
        *exprs: IntoArrowExpr,
        **named_exprs: IntoArrowExpr,
    ) -> Self:
        new_columns = evaluate_into_exprs(self, *exprs, **named_exprs)
        new_column_name_to_new_column_map = {s.name: s for s in new_columns}
        to_concat = []
        output_names = []
        # Make sure to preserve column order
        length = len(self)
        for name in self.columns:
            if name in new_column_name_to_new_column_map:
                to_concat.append(
                    validate_dataframe_comparand(
                        length=length,
                        other=new_column_name_to_new_column_map.pop(name),
                        backend_version=self._backend_version,
                    )
                )
            else:
                to_concat.append(self._native_dataframe[name])
            output_names.append(name)
        for s in new_column_name_to_new_column_map:
            to_concat.append(
                validate_dataframe_comparand(
                    length=length,
                    other=new_column_name_to_new_column_map[s],
                    backend_version=self._backend_version,
                )
            )
            output_names.append(s)
        df = self._native_dataframe.__class__.from_arrays(to_concat, names=output_names)
        return self._from_native_dataframe(df)

    def group_by(self, *keys: str) -> ArrowGroupBy:
        from narwhals._arrow.group_by import ArrowGroupBy

        return ArrowGroupBy(self, list(keys))

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

        how_to_join_map = {
            "anti": "left anti",
            "semi": "left semi",
            "inner": "inner",
            "left": "left outer",
        }

        if how == "cross":
            plx = self.__narwhals_namespace__()
            key_token = generate_unique_token(
                n_bytes=8, columns=[*self.columns, *other.columns]
            )

            return self._from_native_dataframe(
                self.with_columns(**{key_token: plx.lit(0, None)})._native_dataframe.join(
                    other.with_columns(**{key_token: plx.lit(0, None)})._native_dataframe,
                    keys=key_token,
                    right_keys=key_token,
                    join_type="inner",
                    right_suffix="_right",
                ),
            ).drop(key_token)

        return self._from_native_dataframe(
            self._native_dataframe.join(
                other._native_dataframe,
                keys=left_on,
                right_keys=right_on,
                join_type=how_to_join_map[how],
                right_suffix="_right",
            ),
        )

    def drop(self, *columns: str) -> Self:
        return self._from_native_dataframe(self._native_dataframe.drop(list(columns)))

    def drop_nulls(self) -> Self:
        return self._from_native_dataframe(self._native_dataframe.drop_null())

    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        flat_keys = flatten([*flatten([by]), *more_by])
        df = self._native_dataframe

        if isinstance(descending, bool):
            order = "descending" if descending else "ascending"
            sorting = [(key, order) for key in flat_keys]
        else:
            sorting = [
                (key, "descending" if is_descending else "ascending")
                for key, is_descending in zip(flat_keys, descending)
            ]
        return self._from_native_dataframe(df.sort_by(sorting=sorting))

    def to_pandas(self) -> Any:
        return self._native_dataframe.to_pandas()

    def to_numpy(self) -> Any:
        import numpy as np

        return np.column_stack([col.to_numpy() for col in self._native_dataframe.columns])

    def to_dict(self, *, as_series: bool) -> Any:
        df = self._native_dataframe

        names_and_values = zip(df.column_names, df.columns)
        if as_series:
            from narwhals._arrow.series import ArrowSeries

            return {
                name: ArrowSeries(col, name=name, backend_version=self._backend_version)
                for name, col in names_and_values
            }
        else:
            return {name: col.to_pylist() for name, col in names_and_values}

    def with_row_index(self, name: str) -> Self:
        pa = get_pyarrow()
        df = self._native_dataframe

        row_indices = pa.array(range(df.num_rows))
        return self._from_native_dataframe(df.append_column(name, row_indices))

    def filter(
        self,
        *predicates: IntoArrowExpr,
    ) -> Self:
        from narwhals._arrow.namespace import ArrowNamespace

        plx = ArrowNamespace(backend_version=self._backend_version)
        expr = plx.all_horizontal(*predicates)
        # Safety: all_horizontal's expression only returns a single column.
        mask = expr._call(self)[0]
        return self._from_native_dataframe(
            self._native_dataframe.filter(mask._native_series)
        )

    def null_count(self) -> Self:
        pa = get_pyarrow()
        df = self._native_dataframe
        names_and_values = zip(df.column_names, df.columns)

        return self._from_native_dataframe(
            pa.table({name: [col.null_count] for name, col in names_and_values})
        )

    def head(self, n: int) -> Self:
        df = self._native_dataframe
        if n >= 0:
            return self._from_native_dataframe(df.slice(0, n))
        else:
            num_rows = df.num_rows
            return self._from_native_dataframe(df.slice(0, max(0, num_rows + n)))

    def tail(self, n: int) -> Self:
        df = self._native_dataframe
        if n >= 0:
            num_rows = df.num_rows
            return self._from_native_dataframe(df.slice(max(0, num_rows - n)))
        else:
            return self._from_native_dataframe(df.slice(abs(n)))

    def lazy(self) -> Self:
        return self

    def collect(self) -> ArrowDataFrame:
        return ArrowDataFrame(
            self._native_dataframe, backend_version=self._backend_version
        )

    def clone(self) -> Self:
        msg = "clone is not yet supported on PyArrow tables"
        raise NotImplementedError(msg)

    def is_empty(self: Self) -> bool:
        return self.shape[0] == 0

    def item(self: Self, row: int | None = None, column: int | str | None = None) -> Any:
        if row is None and column is None:
            if self.shape != (1, 1):
                msg = (
                    "can only call `.item()` if the dataframe is of shape (1, 1),"
                    " or if explicit row/col values are provided;"
                    f" frame has shape {self.shape!r}"
                )
                raise ValueError(msg)
            return self._native_dataframe[0][0]

        elif row is None or column is None:
            msg = "cannot call `.item()` with only one of `row` or `column`"
            raise ValueError(msg)

        _col = self.columns.index(column) if isinstance(column, str) else column
        return self._native_dataframe[_col][row]

    def rename(self, mapping: dict[str, str]) -> Self:
        df = self._native_dataframe
        new_cols = [mapping.get(c, c) for c in df.column_names]
        return self._from_native_dataframe(df.rename_columns(new_cols))

    def write_parquet(self, file: Any) -> Any:
        pp = get_pyarrow_parquet()
        pp.write_table(self._native_dataframe, file)

    def is_duplicated(self: Self) -> ArrowSeries:
        from narwhals._arrow.series import ArrowSeries

        np = get_numpy()
        pa = get_pyarrow()
        pc = get_pyarrow_compute()
        df = self._native_dataframe

        columns = self.columns
        col_token = generate_unique_token(n_bytes=8, columns=columns)
        row_count = (
            df.append_column(col_token, pa.array(np.arange(len(self))))
            .group_by(columns)
            .aggregate([(col_token, "count")])
        )
        is_duplicated = pc.greater(
            df.join(
                row_count, keys=columns, right_keys=columns, join_type="inner"
            ).column(f"{col_token}_count"),
            1,
        )
        return ArrowSeries(is_duplicated, name="", backend_version=self._backend_version)

    def is_unique(self: Self) -> ArrowSeries:
        from narwhals._arrow.series import ArrowSeries

        pc = get_pyarrow_compute()
        is_duplicated = self.is_duplicated()._native_series

        return ArrowSeries(
            pc.invert(is_duplicated), name="", backend_version=self._backend_version
        )

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

        np = get_numpy()
        pa = get_pyarrow()
        pc = get_pyarrow_compute()

        df = self._native_dataframe

        if isinstance(subset, str):
            subset = [subset]
        subset = subset or self.columns

        if keep in {"any", "first", "last"}:
            agg_func_map = {"any": "min", "first": "min", "last": "max"}

            agg_func = agg_func_map[keep]
            col_token = generate_unique_token(n_bytes=8, columns=self.columns)
            keep_idx = (
                df.append_column(col_token, pa.array(np.arange(len(self))))
                .group_by(subset)
                .aggregate([(col_token, agg_func)])
                .column(f"{col_token}_{agg_func}")
            )

            return self._from_native_dataframe(pc.take(df, keep_idx))

        keep_idx = self.select(*subset).is_unique()
        return self.filter(keep_idx)

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        return self._from_native_dataframe(self._native_dataframe[offset::n])
