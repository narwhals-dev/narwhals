from __future__ import annotations

import operator
import random
from collections import Counter
from collections.abc import Mapping
from itertools import chain, compress, repeat
from typing import TYPE_CHECKING, Any, Literal, overload

from narwhals._compliant import EagerDataFrame
from narwhals._typing_compat import assert_never
from narwhals._utils import (
    Implementation,
    check_column_names_are_unique,
    not_implemented,
    scale_bytes,
)
from narwhals.exceptions import InvalidOperationError, ShapeError
from narwhals_dict.series import DictSeries
from narwhals_dict.utils import is_native_frame

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator, Sequence
    from io import BytesIO
    from pathlib import Path
    from types import ModuleType

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from _typeshed import SupportsWrite
    from typing_extensions import Self, TypeIs

    from narwhals._compliant.typing import CompliantDataFrameAny
    from narwhals._translate import IntoArrowTable
    from narwhals._utils import Version, _LimitedContext
    from narwhals.dtypes import DType
    from narwhals.typing import (
        IntoSchema,
        JoinStrategy,
        SizedMultiIndexSelector,
        SizedMultiNameSelector,
        SizeUnit,
        _2DArray,
        _SliceIndex,
        _SliceName,
    )
    from narwhals_dict.expr import DictExpr
    from narwhals_dict.group_by import DictGroupBy
    from narwhals_dict.namespace import DictNamespace
    from narwhals_dict.typing import DictFrame, NativeSeries


class DictDataFrame(
    EagerDataFrame["DictSeries", "DictExpr", "DictFrame", "NativeSeries"]  # type: ignore[type-var]
):
    _implementation = Implementation.UNKNOWN

    def __init__(
        self,
        native_dataframe: DictFrame,
        *,
        version: Version,
        validate_column_names: bool = True,
        validate_lengths: bool = True,
    ) -> None:
        if validate_column_names:
            check_column_names_are_unique(native_dataframe.keys())
        if validate_lengths:
            lengths = {len(column) for column in native_dataframe.values()}
            if len(lengths) > 1:
                msg = f"Expected all columns to have the same length, got: {sorted(lengths)}."
                raise ShapeError(msg)
        self._native_frame: DictFrame = native_dataframe
        self._version = version

    @staticmethod
    def _is_native(obj: DictFrame | Any) -> TypeIs[DictFrame]:
        return is_native_frame(obj)

    @classmethod
    def from_native(cls, data: DictFrame, /, *, context: _LimitedContext) -> Self:
        return cls(data, version=context._version)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        /,
        *,
        context: _LimitedContext,
        schema: IntoSchema | Mapping[str, DType | None] | None,
    ) -> Self:
        names = list(schema) if schema else list(data)
        native: DictFrame = {name: list(data[name]) if data else [] for name in names}
        result = cls(native, version=context._version)
        if schema:
            casts = {name: dtype for name, dtype in schema.items() if dtype is not None}
            if casts:
                result = result.with_columns(
                    *(
                        result.__narwhals_namespace__().col(name).cast(dtype)
                        for name, dtype in casts.items()
                    )
                )
        return result

    @classmethod
    def from_dicts(
        cls,
        data: Sequence[Mapping[str, Any]],
        /,
        *,
        context: _LimitedContext,
        schema: IntoSchema | Mapping[str, DType | None] | None,
    ) -> Self:
        names = list(schema) if schema else (list(data[0]) if data else [])
        native: DictFrame = {name: [row.get(name) for row in data] for name in names}
        return cls(native, version=context._version)

    @classmethod
    def from_numpy(
        cls,
        data: _2DArray,
        /,
        *,
        context: _LimitedContext,
        schema: IntoSchema | Sequence[str] | None,
    ) -> Self:
        names = (
            list(schema)
            if isinstance(schema, Mapping)
            else cls._numpy_column_names(data, schema)
        )
        native = {
            name: column.tolist() for name, column in zip(names, data.T, strict=True)
        }
        return cls(native, version=context._version)

    # NOTE: Requires https://github.com/narwhals-dev/narwhals/pull/3753
    @classmethod
    def from_arrow(cls, data: IntoArrowTable, /, *, context: _LimitedContext) -> Self:
        from narwhals._utils import _into_arrow_table

        # Reuse narwhals' pyarrow bridge (handles both `pa.Table` and any
        # `__arrow_c_stream__` object), then materialize columns as plain lists.
        table = _into_arrow_table(data, context)
        native: DictFrame = {
            name: column.to_pylist()
            for name, column in zip(table.column_names, table.columns, strict=True)
        }
        return cls(native, version=context._version)

    def __narwhals_namespace__(self) -> DictNamespace:
        from narwhals_dict.namespace import DictNamespace

        return DictNamespace(version=self._version)

    def __native_namespace__(self) -> ModuleType:
        import builtins

        return builtins

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def _with_version(self, version: Version) -> Self:
        return self.__class__(
            self.native,
            version=version,
            validate_column_names=False,
            validate_lengths=False,
        )

    def _with_native(self, df: DictFrame, *, validate_column_names: bool = True) -> Self:
        return self.__class__(
            df,
            version=self._version,
            validate_column_names=validate_column_names,
            validate_lengths=False,
        )

    @property
    def columns(self) -> list[str]:
        return list(self.native.keys())

    @property
    def schema(self) -> dict[str, DType]:
        return {name: self.get_column(name).dtype for name in self.native}

    def collect_schema(self) -> dict[str, DType]:
        return self.schema

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), len(self.native))

    def __len__(self) -> int:
        return len(next(iter(self.native.values()), []))

    # Selection and projection
    def get_column(self, name: str) -> DictSeries:
        if error := self._check_columns_exist([name]):
            raise error
        return DictSeries(self.native[name], name=name, version=self._version)

    def iter_columns(self) -> Iterator[DictSeries]:
        for name, column in self.native.items():
            yield DictSeries(column, name=name, version=self._version)

    _iter_columns = iter_columns

    def simple_select(self, *column_names: str) -> Self:
        if error := self._check_columns_exist(column_names):
            raise error
        return self._with_native(
            {name: self.native[name] for name in column_names},
            validate_column_names=False,
        )

    def select(self, *exprs: DictExpr) -> Self:
        new_series = self._evaluate_exprs(*exprs)
        if not new_series:
            return self._with_native({}, validate_column_names=False)
        names = [s.name for s in new_series]
        check_column_names_are_unique(names)
        aligned = new_series[0]._align_full_broadcast(*new_series)
        return self._with_native(
            {s.name: s.native for s in aligned}, validate_column_names=False
        )

    def with_columns(self, *exprs: DictExpr) -> Self:
        native = dict(self.native)
        for series in self._evaluate_exprs(*exprs):
            native[series.name] = self._extract_comparand(series)
        return self._with_native(native, validate_column_names=False)

    def _extract_comparand(self, other: DictSeries) -> list[Any]:
        length = len(self)
        if not other._broadcast:
            if (len_other := len(other)) != length:
                msg = f"Expected object of length {length}, got: {len_other}."
                raise ShapeError(msg)
            return other.native
        value = other.native[0] if other.native else None
        return [value] * length

    def rename(self, mapping: Mapping[str, str]) -> Self:
        if error := self._check_columns_exist(list(mapping)):
            raise error
        native = {mapping.get(name, name): column for name, column in self.native.items()}
        check_column_names_are_unique(native.keys())
        return self._with_native(native, validate_column_names=False)

    def drop(self, columns: Sequence[str], *, strict: bool) -> Self:
        if strict and (error := self._check_columns_exist(columns)):
            raise error
        to_drop = set(columns)
        return self._with_native(
            {name: column for name, column in self.native.items() if name not in to_drop},
            validate_column_names=False,
        )

    # Row-wise operations
    def _mask_rows(self, mask: Sequence[bool]) -> Self:
        if len(mask) != len(self):
            msg = f"Expected mask of length {len(self)}, got: {len(mask)}."
            raise ShapeError(msg)
        return self._with_native(
            {name: list(compress(column, mask)) for name, column in self.native.items()},
            validate_column_names=False,
        )

    def filter(self, predicate: DictExpr | Any) -> Self:
        if isinstance(predicate, list):
            return self._mask_rows(predicate)
        mask = self._evaluate_single_output_expr(predicate)
        return self._mask_rows(self._extract_comparand(mask))

    def drop_nulls(self, subset: Sequence[str] | None) -> Self:
        if subset is not None and (error := self._check_columns_exist(subset)):
            raise error
        columns = [self.native[name] for name in (subset or self.columns)]
        mask = [
            all(value is not None for value in row) for row in zip(*columns, strict=True)
        ]
        return self._mask_rows(mask) if columns else self

    def _gather(self, rows: SizedMultiIndexSelector[NativeSeries]) -> Self:
        indices = tuple(operator.index(i) for i in rows)
        return self._with_native(
            {name: [column[i] for i in indices] for name, column in self.native.items()},
            validate_column_names=False,
        )

    def _gather_slice(self, rows: _SliceIndex | range) -> Self:
        return self._with_native(
            {
                name: column[rows.start : rows.stop : rows.step]
                for name, column in self.native.items()
            },
            validate_column_names=False,
        )

    def _select_multi_index(self, columns: SizedMultiIndexSelector[NativeSeries]) -> Self:
        names = self.columns
        return self.simple_select(*(names[i] for i in columns))

    def _select_multi_name(self, columns: SizedMultiNameSelector[NativeSeries]) -> Self:
        return self.simple_select(*columns)

    def _select_slice_index(self, columns: _SliceIndex | range) -> Self:
        return self.simple_select(
            *self.columns[columns.start : columns.stop : columns.step]
        )

    def _select_slice_name(self, columns: _SliceName) -> Self:
        names = self.columns
        start = names.index(columns.start) if columns.start is not None else None
        stop = names.index(columns.stop) + 1 if columns.stop is not None else None
        step: Any = columns.step
        return self.simple_select(*names[start:stop:step])

    def head(self, n: int) -> Self:
        return self._gather_slice(slice(None, n))

    def tail(self, n: int) -> Self:
        length = len(self)
        start = max(length - n, 0) if n >= 0 else -n
        return self._gather_slice(slice(start, None))

    def gather_every(self, n: int, offset: int) -> Self:
        return self._gather_slice(slice(offset, None, n))

    def _sorted_indices(
        self, by: Sequence[str], *, descending: bool | Sequence[bool], nulls_last: bool
    ) -> list[int]:
        """Row indices ordered by `by`, ascending unless `descending` says otherwise."""
        columns = [self.native[name] for name in by]
        if isinstance(descending, bool):
            # Uniform direction: sort once on a per-row tuple key, letting Python's
            # tuple comparison do the work. A null can't be in the key directly
            # as `None < value` raises, so each cell becomes a `(rank, value)` pair:
            # non-nulls share rank 0 and compare by value, while nulls take a rank
            # that floats them to the required end and a constant value
            # (so null-vs-null never compares).
            # `reverse` flips the null rank too, hence `nulls_last == descending`.
            null_rank = -1 if nulls_last == descending else 1

            def key(index: int) -> tuple[tuple[int, Any], ...]:
                return tuple(
                    (0, value) if (value := column[index]) is not None else (null_rank, 0)
                    for column in columns
                )

            return sorted(range(len(self)), key=key, reverse=descending)
        # Mixed per-column directions can't be a single tuple sort, so fall back
        # to repeated stable sorts, from the least to the most significant key.
        indices = list(range(len(self)))
        for column, desc in reversed(tuple(zip(columns, descending, strict=True))):
            nulls = [i for i in indices if column[i] is None]
            rest = sorted(
                (i for i in indices if column[i] is not None),
                key=lambda i: column[i],
                reverse=desc,
            )
            indices = rest + nulls if nulls_last else nulls + rest
        return indices

    def sort(self, *by: str, descending: bool | Sequence[bool], nulls_last: bool) -> Self:
        if error := self._check_columns_exist(by):
            raise error
        return self._gather(
            self._sorted_indices(by, descending=descending, nulls_last=nulls_last)
        )

    def top_k(self, k: int, *, by: Iterable[str], reverse: bool | Sequence[bool]) -> Self:
        # TODO(FBruzzesi): Can we do better than sort + head?
        descending = (
            not reverse if isinstance(reverse, bool) else [not flag for flag in reverse]
        )
        return self.sort(*by, descending=descending, nulls_last=True).head(k)

    def unique(
        self,
        subset: Sequence[str] | None,
        *,
        keep: Any,
        maintain_order: bool | None = None,
        order_by: Sequence[str] | None = None,
    ) -> Self:
        if order_by and (error := self._check_columns_exist(order_by)):
            raise error
        if subset is not None and (error := self._check_columns_exist(subset)):
            raise error
        columns = [self.native[name] for name in (subset or self.columns)]
        rows = list(zip(*columns, strict=True))
        if keep in {"any", "first", "last"}:
            # With `order_by`, "first"/"last" are defined by the order the rows
            # take under the order-by columns (ascending, nulls first), not the
            # frame order. Kept rows are still emitted in original order: that is
            # what `maintain_order` expects, and lazy callers sort afterwards.
            order = (
                self._sorted_indices(order_by, descending=False, nulls_last=False)
                if order_by
                else range(len(self))
            )
            seen: dict[Any, int] = {}
            for index in order if keep != "last" else reversed(list(order)):
                seen.setdefault(rows[index], index)
            indices = sorted(seen.values())
        elif keep == "none":
            counts = Counter(rows)
            indices = [i for i, row in enumerate(rows) if counts[row] == 1]
        else:  # pragma: no cover
            msg = f"Unsupported `keep` strategy: {keep}"
            raise ValueError(msg)
        return self._gather(indices)

    def with_row_index(self, name: str, order_by: Sequence[str] | None) -> Self:
        if not order_by:
            index_column: list[Any] = list(range(len(self)))
        else:
            if error := self._check_columns_exist(order_by):
                raise error
            # Number the rows in order-by order, then scatter each rank back to
            # its original position so the index column aligns with the frame.
            index_column = [0] * len(self)
            order = self._sorted_indices(order_by, descending=False, nulls_last=False)
            for rank, original in enumerate(order):
                index_column[original] = rank
        return self._with_native({name: index_column, **self.native})

    # Conversions
    def clone(self) -> Self:
        return self._with_native(
            {name: list(column) for name, column in self.native.items()},
            validate_column_names=False,
        )

    @overload
    def to_dict(self, *, as_series: Literal[True]) -> dict[str, DictSeries]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    def to_dict(self, *, as_series: bool) -> dict[str, DictSeries] | dict[str, list[Any]]:
        if as_series:
            return {series.name: series for series in self.iter_columns()}
        return {name: list(column) for name, column in self.native.items()}

    @overload
    def rows(self, *, named: Literal[True]) -> list[dict[str, Any]]: ...
    @overload
    def rows(self, *, named: Literal[False]) -> list[tuple[Any, ...]]: ...
    @overload
    def rows(self, *, named: bool) -> list[tuple[Any, ...]] | list[dict[str, Any]]: ...
    def rows(self, *, named: bool) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        if named:
            return list(self.iter_rows(named=True, buffer_size=512))
        return list(self.iter_rows(named=False, buffer_size=512))

    @overload
    def iter_rows(
        self, *, named: Literal[True], buffer_size: int
    ) -> Iterator[dict[str, Any]]: ...
    @overload
    def iter_rows(
        self, *, named: Literal[False], buffer_size: int
    ) -> Iterator[tuple[Any, ...]]: ...
    @overload
    def iter_rows(
        self, *, named: bool, buffer_size: int
    ) -> Iterator[tuple[Any, ...]] | Iterator[dict[str, Any]]: ...
    def iter_rows(
        self, *, named: bool, buffer_size: int
    ) -> Iterator[tuple[Any, ...]] | Iterator[dict[str, Any]]:
        # Columns are sliced one chunk of `buffer_size` rows at a time, so extra
        # memory is bounded by the buffer and each chunk is a snapshot: mutating
        # the native lists mid-iteration cannot affect already-buffered rows.
        columns = self.native.values()
        names = self.columns
        num_rows = len(self)

        for idx in range(0, num_rows, buffer_size):
            buffer = (column[idx : idx + buffer_size] for column in columns)
            if named:
                for row in zip(*buffer, strict=True):
                    yield dict(zip(names, row, strict=True))
            else:
                yield from zip(*buffer, strict=True)

    def row(self, index: int) -> tuple[Any, ...]:
        return tuple(column[index] for column in self.native.values())

    def item(self, row: int | None, column: int | str | None) -> Any:
        if row is None and column is None:
            if self.shape != (1, 1):
                msg = (
                    'can only call `.item()` without "row" or "column" values '
                    f"if the DataFrame has a single element; shape={self.shape!r}"
                )
                raise ValueError(msg)
            return next(iter(self.native.values()))[0]
        if row is None or column is None:
            msg = "cannot call `.item()` with only one of `row` or `column`"
            raise ValueError(msg)
        name = self.columns[column] if isinstance(column, int) else column
        return self.native[name][row]

    def group_by(
        self, keys: Sequence[str] | Sequence[DictExpr], *, drop_null_keys: bool
    ) -> DictGroupBy:
        from narwhals_dict.group_by import DictGroupBy

        return DictGroupBy(self, keys, drop_null_keys=drop_null_keys)

    # Joins are plain hash joins: build `key -> row indices` for the right side,
    # probe with the left rows, then gather both sides into the output columns.
    # Following polars semantics: null keys do not match (nulls_equal = False is
    # the polars default), `inner`/`left` coalesce the key columns (right keys are
    # dropped), `full` keeps both sides' keys, and right columns colliding with
    # left ones get the suffix appended.
    @staticmethod
    def _take(column: NativeSeries, indices: Sequence[Any]) -> list[Any]:
        """Gather values at `indices`, which must not contain `None`."""
        return list(map(column.__getitem__, indices))

    @staticmethod
    def _take_nullable(column: NativeSeries, indices: Sequence[Any]) -> list[Any]:
        """Gather values at `indices`, where a `None` index yields a null."""
        return [None if i is None else column[i] for i in indices]

    def _iter_join_keys(self, keys: Sequence[str]) -> Iterator[Any]:
        """Yields one hashable key per row, or `None` for rows with any null key.

        Single-key joins use the column values directly (no tuple per row);
        multi-key joins yield the value tuple. `None in row` checks identity
        first, so it only falls back to `==` for values that are not nulls.
        """
        if len(keys) == 1:
            yield from self.native[keys[0]]
        else:
            for row in self.simple_select(*keys).iter_rows(named=False, buffer_size=512):
                yield None if None in row else row

    def _join_key_index(self, keys: Sequence[str]) -> dict[Any, list[int]]:
        """Map each (non-null) key to the row indices where it appears."""
        index: dict[Any, list[int]] = {}
        for row, key in enumerate(self._iter_join_keys(keys)):
            if key is not None:
                index.setdefault(key, []).append(row)
        return index

    def _join_key_set(self, keys: Sequence[str]) -> set[Any]:
        """Distinct non-null keys; enough for `semi`/`anti`, cheaper than the index."""
        key_set = set(self._iter_join_keys(keys))
        key_set.discard(None)
        return key_set

    def _join_output_names(
        self, other: Self, *, exclude_right: Collection[str], suffix: str
    ) -> dict[str, str]:
        """Map right column names to output names, raising on duplicates."""
        left_names = set(self.columns)
        right_names = {
            name: f"{name}{suffix}" if name in left_names else name
            for name in other.columns
            if name not in exclude_right
        }
        check_column_names_are_unique([*self.columns, *right_names.values()])
        return right_names

    def _join_gather(
        self,
        other: Self,
        left_indices: Sequence[int | None],
        right_indices: Sequence[int | None],
        *,
        exclude_right: Collection[str],
        suffix: str,
    ) -> Self:
        right_names = self._join_output_names(
            other, exclude_right=exclude_right, suffix=suffix
        )
        # `None in list` is a single identity-first C-speed scan; paying it once
        # per side buys the fast `map` gather for every column of that side.
        take_left = self._take_nullable if None in left_indices else self._take
        take_right = self._take_nullable if None in right_indices else self._take
        result: DictFrame = {
            name: take_left(column, left_indices) for name, column in self.native.items()
        }
        for name, output_name in right_names.items():
            result[output_name] = take_right(other.native[name], right_indices)
        return self._with_native(result, validate_column_names=False)

    def _join_inner(
        self, other: Self, *, left_on: Sequence[str], right_on: Sequence[str], suffix: str
    ) -> Self:
        right_index = other._join_key_index(right_on)
        left_indices: list[int | None] = []
        right_indices: list[int | None] = []
        get_matches = right_index.get
        for row, key in enumerate(self._iter_join_keys(left_on)):
            if key is not None and (matches := get_matches(key)) is not None:
                left_indices.extend(repeat(row, len(matches)))
                right_indices.extend(matches)
        return self._join_gather(
            other, left_indices, right_indices, exclude_right=set(right_on), suffix=suffix
        )

    def _join_left(
        self, other: Self, *, left_on: Sequence[str], right_on: Sequence[str], suffix: str
    ) -> Self:
        right_index = other._join_key_index(right_on)
        left_indices: list[int | None] = []
        right_indices: list[int | None] = []
        get_matches = right_index.get
        for row, key in enumerate(self._iter_join_keys(left_on)):
            matches = get_matches(key) if key is not None else None
            if matches:
                left_indices.extend(repeat(row, len(matches)))
                right_indices.extend(matches)
            else:
                left_indices.append(row)
                right_indices.append(None)
        return self._join_gather(
            other, left_indices, right_indices, exclude_right=set(right_on), suffix=suffix
        )

    def _join_full(
        self, other: Self, *, left_on: Sequence[str], right_on: Sequence[str], suffix: str
    ) -> Self:
        # Keys are not coalesced: both sides' keys stay in the output, and rows
        # unmatched on either side get nulls for the opposite side's columns.
        right_index = other._join_key_index(right_on)
        left_indices: list[int | None] = []
        right_indices: list[int | None] = []
        matched_right: set[int] = set()
        get_matches = right_index.get
        for row, key in enumerate(self._iter_join_keys(left_on)):
            matches = get_matches(key) if key is not None else None
            if matches:
                left_indices.extend(repeat(row, len(matches)))
                right_indices.extend(matches)
                matched_right.update(matches)
            else:
                left_indices.append(row)
                right_indices.append(None)
        for row in range(len(other)):
            if row not in matched_right:
                left_indices.append(None)
                right_indices.append(row)
        return self._join_gather(
            other, left_indices, right_indices, exclude_right=(), suffix=suffix
        )

    def _join_cross(self, other: Self, *, suffix: str) -> Self:
        # Values are laid out directly, no index lists: each left value is
        # repeated `len(other)` times and each right column is tiled `len(self)`
        # times, all at C speed via `chain`/`repeat` and list multiplication.
        right_names = self._join_output_names(other, exclude_right=(), suffix=suffix)
        n_left, n_right = len(self), len(other)
        result: DictFrame = {
            name: list(chain.from_iterable(map(repeat, column, repeat(n_right))))
            for name, column in self.native.items()
        }
        for name, output_name in right_names.items():
            result[output_name] = list(other.native[name]) * n_left
        return self._with_native(result, validate_column_names=False)

    def _join_semi(
        self, other: Self, *, left_on: Sequence[str], right_on: Sequence[str]
    ) -> Self:
        right_keys = other._join_key_set(right_on)
        return self._mask_rows(
            [key in right_keys for key in self._iter_join_keys(left_on)]
        )

    def _join_anti(
        self, other: Self, *, left_on: Sequence[str], right_on: Sequence[str]
    ) -> Self:
        right_keys = other._join_key_set(right_on)
        return self._mask_rows(
            [key not in right_keys for key in self._iter_join_keys(left_on)]
        )

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
            return self._join_cross(other, suffix=suffix)
        if left_on is None or right_on is None:  # pragma: no cover
            raise ValueError(left_on, right_on)
        if error := self._check_columns_exist(left_on):
            raise error
        if error := other._check_columns_exist(right_on):
            raise error
        if how == "inner":
            return self._join_inner(
                other, left_on=left_on, right_on=right_on, suffix=suffix
            )
        if how == "left":
            return self._join_left(
                other, left_on=left_on, right_on=right_on, suffix=suffix
            )
        if how == "full":
            return self._join_full(
                other, left_on=left_on, right_on=right_on, suffix=suffix
            )
        if how == "semi":
            return self._join_semi(other, left_on=left_on, right_on=right_on)
        if how == "anti":
            return self._join_anti(other, left_on=left_on, right_on=right_on)
        assert_never(how)

    def lazy(self, backend: Any = None, *, session: Any = None) -> Any:
        if backend is None:
            return self
        msg = "`lazy` with a backend is not supported for the dict backend."
        raise NotImplementedError(msg)

    def collect(self, backend: Any, **kwargs: Any) -> CompliantDataFrameAny:
        if backend is None:
            return self
        msg = "`collect` with a backend is not supported for the dict backend."
        raise NotImplementedError(msg)

    def is_unique(self) -> DictSeries:
        rows = list(zip(*self.native.values(), strict=True))
        counts = Counter(rows)
        return DictSeries(
            [counts[row] == 1 for row in rows], name="", version=self._version
        )

    def to_arrow(self) -> pa.Table:
        import pyarrow as pa  # ignore-banned-import

        return pa.Table.from_pydict(
            {name: list(column) for name, column in self.native.items()}
        )

    def to_pandas(self) -> pd.DataFrame:
        import pandas as pd  # ignore-banned-import

        return pd.DataFrame(dict(self.native))

    def to_polars(self) -> pl.DataFrame:
        import polars as pl  # ignore-banned-import

        return pl.DataFrame(dict(self.native))

    def __array__(self, dtype: Any = None, *, copy: bool | None = None) -> _2DArray:
        import numpy as np  # ignore-banned-import

        # Convert each column through the Series so time-zone-aware datetimes
        # (-> UTC, naive `datetime64`) and numeric-with-null columns (-> float64
        # NaN) are handled once, then stack the columns side by side.
        columns = [series.__array__(dtype, copy=copy) for series in self.iter_columns()]
        if not columns:
            return np.empty((len(self), 0), dtype=dtype)
        return np.column_stack(columns)

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _2DArray:
        return self.__array__(dtype, copy=copy)

    def estimated_size(self, unit: SizeUnit) -> int | float:
        from sys import getsizeof

        # TODO(FBruzzesi): Is this the right/best approach?
        total = sum(
            getsizeof(column) + sum(map(getsizeof, column))
            for column in self.native.values()
        )
        return scale_bytes(total, unit)

    @overload
    def write_csv(self, file: None) -> str: ...
    @overload
    def write_csv(self, file: str | Path | BytesIO) -> None: ...
    def write_csv(self, file: str | Path | BytesIO | None = None) -> str | None:
        import csv
        from io import StringIO
        from pathlib import Path

        def dump(stream: SupportsWrite[str]) -> None:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(self.columns)
            writer.writerows(self.iter_rows(named=False, buffer_size=512))

        if file is None:
            buffer = StringIO()
            dump(buffer)
            return buffer.getvalue()
        if isinstance(file, (str, Path)):
            with Path(file).open("w", newline="", encoding="utf-8") as stream:
                dump(stream)
        else:  # binary stream, e.g. `BytesIO`
            buffer = StringIO()
            dump(buffer)
            file.write(buffer.getvalue().encode())
        return None

    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self:
        index = [] if index is None else list(index)
        on = [c for c in self.columns if c not in index] if on is None else list(on)
        if error := self._check_columns_exist([*index, *on]):
            raise error
        n = len(self)
        native = self.native

        result: DictFrame = {name: list(native[name]) * len(on) for name in index}
        result[variable_name] = list(chain.from_iterable(repeat(name, n) for name in on))
        result[value_name] = list(chain.from_iterable(native[name] for name in on))
        return self._with_native(result)

    def explode(self, columns: Sequence[str]) -> Self:
        dtypes = self._version.dtypes
        for name in columns:
            if not isinstance((dtype := self.get_column(name).dtype), dtypes.List):
                msg = (
                    f"`explode` operation not supported for dtype `{dtype}`, "
                    "expected List type"
                )
                raise InvalidOperationError(msg)
        # Per-row element count of the first exploded column, the anchor every
        # other exploded column must match. A null list counts as `None` (its own
        # value, distinct from any int) so it only matches another null, while an
        # empty list counts as `0` -- both still explode to a single null row.
        counts = [None if cell is None else len(cell) for cell in self.native[columns[0]]]
        to_explode = set(columns)
        for name in columns[1:]:
            column = self.native[name]
            if any(
                (None if cell is None else len(cell)) != count
                for cell, count in zip(column, counts, strict=True)
            ):
                msg = "exploded columns must have matching element counts"
                raise ShapeError(msg)
        result: DictFrame = {}
        for name, column in self.native.items():
            if name in to_explode:
                # `if cell:` is False for both `None` and `[]`, which each yield a
                # single null; a non-empty list contributes its elements as-is.
                result[name] = [
                    element for cell in column for element in (cell or (None,))
                ]
            else:
                result[name] = [
                    value
                    for value, count in zip(column, counts, strict=True)
                    for _ in range(count or 1)
                ]
        return self._with_native(result, validate_column_names=False)

    def sample(self, n: int, *, with_replacement: bool, seed: int | None) -> Self:
        rng = random.Random(seed)  # noqa: S311
        population = range(len(self))
        indices = (
            rng.choices(population, k=n)
            if with_replacement
            else rng.sample(population, k=n)
        )
        return self._gather(indices)

    # Not implemented (yet): fill in incrementally.
    join_asof = not_implemented()
    pivot = not_implemented()
    write_parquet = not_implemented()
