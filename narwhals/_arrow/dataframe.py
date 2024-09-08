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
from narwhals.dependencies import get_pyarrow
from narwhals.dependencies import is_numpy_array
from narwhals.utils import Implementation
from narwhals.utils import flatten
from narwhals.utils import generate_unique_token
from narwhals.utils import parse_columns_to_drop

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._arrow.group_by import ArrowGroupBy
    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals._arrow.series import ArrowSeries
    from narwhals._arrow.typing import IntoArrowExpr
    from narwhals.dtypes import DType


class ArrowDataFrame:
    # --- not in the spec ---
    def __init__(
        self, native_dataframe: pa.Table, *, backend_version: tuple[int, ...]
    ) -> None:
        self._native_frame = native_dataframe
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

    def _from_native_frame(self, df: Any) -> Self:
        return self.__class__(df, backend_version=self._backend_version)

    @property
    def shape(self) -> tuple[int, int]:
        return self._native_frame.shape  # type: ignore[no-any-return]

    def __len__(self) -> int:
        return len(self._native_frame)

    def row(self, index: int) -> tuple[Any, ...]:
        return tuple(col[index] for col in self._native_frame)

    def rows(
        self, *, named: bool = False
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        if not named:
            msg = "Unnamed rows are not yet supported on PyArrow tables"
            raise NotImplementedError(msg)
        return self._native_frame.to_pylist()  # type: ignore[no-any-return]

    def iter_rows(
        self,
        *,
        named: bool = False,
        buffer_size: int = 512,
    ) -> Iterator[tuple[Any, ...]] | Iterator[dict[str, Any]]:
        df = self._native_frame
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
            self._native_frame[name],
            name=name,
            backend_version=self._backend_version,
        )

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        return self._native_frame.__array__(dtype, copy=copy)

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
                self._native_frame[item],
                name=item,
                backend_version=self._backend_version,
            )
        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[1], (list, tuple))
        ):
            if item[0] == slice(None):
                selected_rows = self._native_frame
            else:
                selected_rows = self._native_frame.take(item[0])

            return self._from_native_frame(selected_rows.select(item[1]))

        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                columns = self.columns
                if isinstance(item[1].start, str) or isinstance(item[1].stop, str):
                    start = (
                        columns.index(item[1].start)
                        if item[1].start is not None
                        else None
                    )
                    stop = (
                        columns.index(item[1].stop) + 1
                        if item[1].stop is not None
                        else None
                    )
                    step = item[1].step
                    return self._from_native_frame(
                        self._native_frame.take(item[0]).select(columns[start:stop:step])
                    )
                if isinstance(item[1].start, int) or isinstance(item[1].stop, int):
                    return self._from_native_frame(
                        self._native_frame.take(item[0]).select(
                            columns[item[1].start : item[1].stop : item[1].step]
                        )
                    )
                msg = f"Expected slice of integers or strings, got: {type(item[1])}"  # pragma: no cover
                raise TypeError(msg)  # pragma: no cover

            from narwhals._arrow.series import ArrowSeries

            # PyArrow columns are always strings
            col_name = item[1] if isinstance(item[1], str) else self.columns[item[1]]
            return ArrowSeries(
                self._native_frame[col_name].take(item[0]),
                name=col_name,
                backend_version=self._backend_version,
            )

        elif isinstance(item, slice):
            if item.step is not None and item.step != 1:
                msg = "Slicing with step is not supported on PyArrow tables"
                raise NotImplementedError(msg)
            start = item.start or 0
            stop = item.stop or len(self._native_frame)
            return self._from_native_frame(
                self._native_frame.slice(item.start, stop - start),
            )

        elif isinstance(item, Sequence) or (is_numpy_array(item) and item.ndim == 1):
            return self._from_native_frame(self._native_frame.take(item))

        else:  # pragma: no cover
            msg = f"Expected str or slice, got: {type(item)}"
            raise TypeError(msg)

    @property
    def schema(self) -> dict[str, DType]:
        schema = self._native_frame.schema
        return {
            name: translate_dtype(dtype)
            for name, dtype in zip(schema.names, schema.types)
        }

    def collect_schema(self) -> dict[str, DType]:
        return self.schema

    @property
    def columns(self) -> list[str]:
        return self._native_frame.schema.names  # type: ignore[no-any-return]

    def select(
        self,
        *exprs: IntoArrowExpr,
        **named_exprs: IntoArrowExpr,
    ) -> Self:
        import pyarrow as pa  # ignore-banned-import()

        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        if not new_series:
            # return empty dataframe, like Polars does
            return self._from_native_frame(self._native_frame.__class__.from_arrays([]))
        names = [s.name for s in new_series]
        df = pa.Table.from_arrays(
            broadcast_series(new_series),
            names=names,
        )
        return self._from_native_frame(df)

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
                to_concat.append(self._native_frame[name])
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
        df = self._native_frame.__class__.from_arrays(to_concat, names=output_names)
        return self._from_native_frame(df)

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

            return self._from_native_frame(
                self.with_columns(**{key_token: plx.lit(0, None)})
                ._native_frame.join(
                    other.with_columns(**{key_token: plx.lit(0, None)})._native_frame,
                    keys=key_token,
                    right_keys=key_token,
                    join_type="inner",
                    right_suffix="_right",
                )
                .drop([key_token]),
            )

        return self._from_native_frame(
            self._native_frame.join(
                other._native_frame,
                keys=left_on,
                right_keys=right_on,
                join_type=how_to_join_map[how],
                right_suffix="_right",
            ),
        )

    def join_asof(
        self,
        other: Self,
        *,
        left_on: str | None = None,
        right_on: str | None = None,
        on: str | None = None,
        by_left: str | list[str] | None = None,
        by_right: str | list[str] | None = None,
        by: str | list[str] | None = None,
        strategy: Literal["backward", "forward", "nearest"] = "backward",
    ) -> Self:
        msg = "join_asof is not yet supported on PyArrow tables"
        raise NotImplementedError(msg)

    def drop(self: Self, columns: list[str], strict: bool) -> Self:  # noqa: FBT001
        to_drop = parse_columns_to_drop(
            compliant_frame=self, columns=columns, strict=strict
        )
        return self._from_native_frame(self._native_frame.drop(to_drop))

    def drop_nulls(self: Self, subset: str | list[str] | None) -> Self:
        if subset is None:
            return self._from_native_frame(self._native_frame.drop_null())
        subset = [subset] if isinstance(subset, str) else subset
        plx = self.__narwhals_namespace__()
        return self.filter(~plx.any_horizontal(plx.col(*subset).is_null()))

    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        flat_keys = flatten([*flatten([by]), *more_by])
        df = self._native_frame

        if isinstance(descending, bool):
            order = "descending" if descending else "ascending"
            sorting = [(key, order) for key in flat_keys]
        else:
            sorting = [
                (key, "descending" if is_descending else "ascending")
                for key, is_descending in zip(flat_keys, descending)
            ]
        return self._from_native_frame(df.sort_by(sorting=sorting))

    def to_pandas(self) -> Any:
        return self._native_frame.to_pandas()

    def to_numpy(self) -> Any:
        import numpy as np  # ignore-banned-import

        return np.column_stack([col.to_numpy() for col in self._native_frame.columns])

    def to_dict(self, *, as_series: bool) -> Any:
        df = self._native_frame

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
        import pyarrow as pa  # ignore-banned-import()

        df = self._native_frame

        row_indices = pa.array(range(df.num_rows))
        return self._from_native_frame(df.append_column(name, row_indices))

    def filter(
        self,
        *predicates: IntoArrowExpr,
    ) -> Self:
        if (
            len(predicates) == 1
            and isinstance(predicates[0], list)
            and all(isinstance(x, bool) for x in predicates[0])
        ):
            mask = predicates[0]
        else:
            plx = self.__narwhals_namespace__()
            expr = plx.all_horizontal(*predicates)
            # Safety: all_horizontal's expression only returns a single column.
            mask = expr._call(self)[0]._native_series
        return self._from_native_frame(self._native_frame.filter(mask))

    def null_count(self) -> Self:
        import pyarrow as pa  # ignore-banned-import()

        df = self._native_frame
        names_and_values = zip(df.column_names, df.columns)

        return self._from_native_frame(
            pa.table({name: [col.null_count] for name, col in names_and_values})
        )

    def head(self, n: int) -> Self:
        df = self._native_frame
        if n >= 0:
            return self._from_native_frame(df.slice(0, n))
        else:
            num_rows = df.num_rows
            return self._from_native_frame(df.slice(0, max(0, num_rows + n)))

    def tail(self, n: int) -> Self:
        df = self._native_frame
        if n >= 0:
            num_rows = df.num_rows
            return self._from_native_frame(df.slice(max(0, num_rows - n)))
        else:
            return self._from_native_frame(df.slice(abs(n)))

    def lazy(self) -> Self:
        return self

    def collect(self) -> ArrowDataFrame:
        return ArrowDataFrame(self._native_frame, backend_version=self._backend_version)

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
            return self._native_frame[0][0]

        elif row is None or column is None:
            msg = "cannot call `.item()` with only one of `row` or `column`"
            raise ValueError(msg)

        _col = self.columns.index(column) if isinstance(column, str) else column
        return self._native_frame[_col][row]

    def rename(self, mapping: dict[str, str]) -> Self:
        df = self._native_frame
        new_cols = [mapping.get(c, c) for c in df.column_names]
        return self._from_native_frame(df.rename_columns(new_cols))

    def write_parquet(self, file: Any) -> Any:
        import pyarrow.parquet as pp  # ignore-banned-import

        pp.write_table(self._native_frame, file)

    def write_csv(self, file: Any) -> Any:
        import pyarrow as pa  # ignore-banned-import
        import pyarrow.csv as pa_csv  # ignore-banned-import

        pa_table = self._native_frame
        if file is None:
            csv_buffer = pa.BufferOutputStream()
            pa_csv.write_csv(pa_table, csv_buffer)
            return csv_buffer.getvalue().to_pybytes().decode()
        return pa_csv.write_csv(pa_table, file)

    def is_duplicated(self: Self) -> ArrowSeries:
        import numpy as np  # ignore-banned-import
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        from narwhals._arrow.series import ArrowSeries

        df = self._native_frame

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
        import pyarrow.compute as pc  # ignore-banned-import()

        from narwhals._arrow.series import ArrowSeries

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
        import numpy as np  # ignore-banned-import
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        df = self._native_frame

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

            return self._from_native_frame(pc.take(df, keep_idx))

        keep_idx = self.select(*subset).is_unique()
        return self.filter(keep_idx)

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        return self._from_native_frame(self._native_frame[offset::n])

    def to_arrow(self: Self) -> Any:
        return self._native_frame
