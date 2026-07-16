from __future__ import annotations

import heapq
import operator
import random
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Mapping
from itertools import chain, compress, islice, product, repeat
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

    from narwhals._compliant.typing import CompliantDataFrameAny, CompliantLazyFrameAny
    from narwhals._spark_like.utils import SparkSession
    from narwhals._translate import IntoArrowTable
    from narwhals._typing import _EagerAllowedImpl, _LazyAllowedImpl
    from narwhals._utils import Version, _LimitedContext
    from narwhals.dtypes import DType
    from narwhals.typing import (
        AsofJoinStrategy,
        IntoDType,
        JoinStrategy,
        PivotAgg,
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
        schema: Mapping[str, IntoDType | None] | None,
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
        schema: Mapping[str, IntoDType | None] | None,
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
        schema: Mapping[str, IntoDType] | Sequence[str] | None,
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
        # Public entry point (e.g. `df[indices]`): coerce arbitrary index objects
        # (NumPy ints, ...) to `int` once, then reuse across every column.
        return self._gather_positions([operator.index(i) for i in rows])

    def _gather_positions(self, indices: Iterable[int]) -> Self:
        """Gather rows at `indices`, which must already be plain `int` positions.

        The trusted-int counterpart to `_gather`: internal callers that build
        their own positions (`sort`, `unique`, `sample`, grouping) skip the
        per-row `operator.index` coercion.
        """
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
        if len(columns) == 1:
            # Single key (the common case): sort the non-null positions directly
            # on the column values and add nulls onto the requested end.
            # Both passes and the stable sort preserve original order for ties.
            column, size = columns[0], len(self)
            desc = descending if isinstance(descending, bool) else descending[0]
            non_null = sorted(
                (i for i in range(size) if column[i] is not None),
                key=column.__getitem__,
                reverse=desc,
            )
            nulls = [i for i in range(size) if column[i] is None]
            return non_null + nulls if nulls_last else nulls + non_null
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
        return self._gather_positions(
            self._sorted_indices(by, descending=descending, nulls_last=nulls_last)
        )

    def top_k(self, k: int, *, by: Iterable[str], reverse: bool | Sequence[bool]) -> Self:
        by = list(by)
        if error := self._check_columns_exist(by):
            raise error
        if len(by) == 1:
            column = self.native[by[0]]
            smallest = reverse if isinstance(reverse, bool) else reverse[0]
            picker = heapq.nsmallest if smallest else heapq.nlargest
            indices = (i for i in range(len(self)) if column[i] is not None)
            top = picker(k, indices, key=column.__getitem__)
            if len(top) < k:
                # `k` exceeds the non-null count, so `sort(nulls_last=True).head(k)`
                # would spill into the null rows (original order); match that.
                nulls = (i for i in range(len(self)) if column[i] is None)
                top += list(islice(nulls, k - len(top)))
            return self._gather_positions(top)
        descending = (
            not reverse if isinstance(reverse, bool) else [not r for r in reverse]
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
        if keep in {"any", "first", "last"}:
            seen: dict[Any, int] = {}
            if order_by:
                rows = list(zip(*columns, strict=True))
                order = self._sorted_indices(order_by, descending=False, nulls_last=False)
                for index in order if keep != "last" else reversed(order):
                    seen.setdefault(rows[index], index)
            elif keep == "last":
                size = len(self)
                reversed_rows = zip(*map(reversed, columns), strict=True)
                for index, row in enumerate(reversed_rows):
                    seen.setdefault(row, size - 1 - index)
            else:
                for index, row in enumerate(zip(*columns, strict=True)):
                    seen.setdefault(row, index)

            indices = sorted(seen.values())
        elif keep == "none":
            counts = Counter(zip(*columns, strict=True))
            indices = [
                index
                for index, row in enumerate(zip(*columns, strict=True))
                if counts[row] == 1
            ]
        else:  # pragma: no cover
            msg = f"Unsupported `keep` strategy: {keep}"
            raise ValueError(msg)
        return self._gather_positions(indices)

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

    def _join_keys(self, keys: Sequence[str]) -> Iterable[Any]:
        """One hashable key per row, with `None` for rows with any null key.

        Single-key joins reuse the column itself (no copy, no tuple per row);
        multi-key joins yield one value tuple per row.
        """
        if len(keys) == 1:
            return self.native[keys[0]]
        columns = [self.native[key] for key in keys]
        return (None if None in row else row for row in zip(*columns, strict=True))

    def _join_key_index(self, keys: Sequence[str]) -> dict[Any, list[int]]:
        """Map each (non-null) key to the row indices where it appears.

        Null keys are left out, so probing with `index.get` makes them miss
        without a per-row null check (`None` is never a stored key).
        """
        index: dict[Any, list[int]] = {}
        for row, key in enumerate(self._join_keys(keys)):
            if key is not None:
                index.setdefault(key, []).append(row)
        return index

    def _join_key_set(self, keys: Sequence[str]) -> set[Any]:
        """Distinct non-null keys; enough for `semi`/`anti`, cheaper than the index."""
        key_set = set(self._join_keys(keys))
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
        left_rows: Sequence[int],
        counts: Sequence[int],
        right_indices: Sequence[int | None],
        *,
        left_tail: int,
        exclude_right: Collection[str],
        suffix: str,
    ) -> Self:
        """Assemble the joined frame from a run-length left selection.

        The output has one row per entry of `right_indices`: first the matched block,
        where left row `left_rows[i]` repeats `counts[i]` times, then `left_tail` rows
        whose left side is all null (unmatched right rows of a full join).
        """
        right_names = self._join_output_names(
            other, exclude_right=exclude_right, suffix=suffix
        )
        # Runs are all length 1 only if the lengths match *and* no run is empty
        # (`counts` may hold zeros for unmatched inner-join rows, and e.g. one 0
        # and one 2 also keep the lengths equal).
        n_left = len(left_rows)
        all_single = (n_left == len(right_indices) - left_tail) and (0 not in counts)
        if all_single and left_tail == 0 and n_left == len(self):
            # Every left row kept exactly once, in order: reuse the columns
            # as-is (native lists are shared freely, never mutated in place).
            result: DictFrame = dict(self.native)
        else:
            left_indices: Sequence[int | None] = (
                left_rows
                if all_single
                else list(chain.from_iterable(map(repeat, left_rows, counts)))
            )
            take_left = self._take
            if left_tail:
                left_indices = (*left_indices, *repeat(None, left_tail))
                take_left = self._take_nullable
            result = {
                name: take_left(column, left_indices)
                for name, column in self.native.items()
            }
        take_right = self._take_nullable if None in right_indices else self._take
        for name, output_name in right_names.items():
            result[output_name] = take_right(other.native[name], right_indices)
        return self._with_native(result, validate_column_names=False)

    def _join_inner(
        self, other: Self, *, left_on: Sequence[str], right_on: Sequence[str], suffix: str
    ) -> Self:
        get_matches = other._join_key_index(right_on).get
        # `get_matches` misses on null keys directly (see `_join_key_index`);
        # row positions stay implicit: `range` plus zero-length runs for misses.
        # `filter(None)` drops the misses; match buckets are never empty, so no
        # falsy value is lost.
        matches_per_row = list(map(get_matches, self._join_keys(left_on)))
        counts = [0 if matches is None else len(matches) for matches in matches_per_row]
        right_indices: list[int | None] = list(
            chain.from_iterable(filter(None, matches_per_row))
        )
        return self._join_gather(
            other,
            range(len(counts)),
            counts,
            right_indices,
            left_tail=0,
            exclude_right=set(right_on),
            suffix=suffix,
        )

    def _join_left(
        self, other: Self, *, left_on: Sequence[str], right_on: Sequence[str], suffix: str
    ) -> Self:
        get_matches = other._join_key_index(right_on).get
        counts: list[int] = []
        right_indices: list[int | None] = []
        add_count = counts.append
        add_null, add_matches = right_indices.append, right_indices.extend
        for matches in map(get_matches, self._join_keys(left_on)):
            if matches is None:
                add_count(1)
                add_null(None)
            else:
                add_count(len(matches))
                add_matches(matches)
        return self._join_gather(
            other,
            range(len(self)),  # every left row is kept, in order
            counts,
            right_indices,
            left_tail=0,
            exclude_right=set(right_on),
            suffix=suffix,
        )

    def _join_full(
        self, other: Self, *, left_on: Sequence[str], right_on: Sequence[str], suffix: str
    ) -> Self:
        # Keys are not coalesced: both sides' keys stay in the output, and rows
        # unmatched on either side get nulls for the opposite side's columns.
        get_matches = other._join_key_index(right_on).get
        counts: list[int] = []
        right_indices: list[int | None] = []
        matched_right: set[int] = set()
        add_count = counts.append
        add_null, add_matches = right_indices.append, right_indices.extend
        for matches in map(get_matches, self._join_keys(left_on)):
            if matches is None:
                add_count(1)
                add_null(None)
            else:
                add_count(len(matches))
                add_matches(matches)
                matched_right.update(matches)
        unmatched_right = [row for row in range(len(other)) if row not in matched_right]
        right_indices.extend(unmatched_right)
        return self._join_gather(
            other,
            range(len(self)),
            counts,
            right_indices,
            left_tail=len(unmatched_right),
            exclude_right=(),
            suffix=suffix,
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
        contains = other._join_key_set(right_on).__contains__
        return self._mask_rows(list(map(contains, self._join_keys(left_on))))

    def _join_anti(
        self, other: Self, *, left_on: Sequence[str], right_on: Sequence[str]
    ) -> Self:
        contains = other._join_key_set(right_on).__contains__
        return self._mask_rows(
            list(map(operator.not_, map(contains, self._join_keys(left_on))))
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

    def _asof_candidates(
        self, on: str, by: Sequence[str] | None
    ) -> dict[Any, tuple[list[Any], list[int]]]:
        """Sorted `(keys, row indices)` per `by` group, for bisecting.

        Rows with a null asof key (or any null `by` key) are never match
        candidates. Keys are sorted here rather than validated as sorted, so
        the probe side needs no ordering assumptions at all.
        """
        groups: dict[Any, list[tuple[Any, int]]] = {}
        by_keys: Iterable[Any] = repeat(None) if by is None else self._join_keys(by)
        for row, (key, by_key) in enumerate(zip(self.native[on], by_keys, strict=False)):
            if key is None or (by is not None and by_key is None):
                continue
            groups.setdefault(by_key, []).append((key, row))
        result: dict[Any, tuple[list[Any], list[int]]] = {}
        for by_key, pairs in groups.items():
            pairs.sort(key=operator.itemgetter(0))
            result[by_key] = ([key for key, _ in pairs], [row for _, row in pairs])
        return result

    def join_asof(
        self,
        other: Self,
        *,
        left_on: str,
        right_on: str,
        by_left: Sequence[str] | None,
        by_right: Sequence[str] | None,
        strategy: AsofJoinStrategy,
        suffix: str,
    ) -> Self:
        if error := self._check_columns_exist([left_on, *(by_left or ())]):
            raise error
        if error := other._check_columns_exist([right_on, *(by_right or ())]):
            raise error
        candidates = other._asof_candidates(right_on, by_right)
        by_keys: Iterable[Any] = (
            repeat(None) if by_left is None else self._join_keys(by_left)
        )
        right_indices: list[int | None] = []
        for key, by_key in zip(self.native[left_on], by_keys, strict=False):
            if (group := candidates.get(by_key) if key is not None else None) is None:
                right_indices.append(None)
                continue
            keys, rows = group
            backward = bisect_right(keys, key) - 1  # last right key <= left key
            forward = bisect_left(keys, key)  # first right key >= left key
            if strategy == "backward":
                match = rows[backward] if backward >= 0 else None
            elif strategy == "forward":
                match = rows[forward] if forward < len(rows) else None
            elif forward >= len(rows):
                match = rows[backward] if backward >= 0 else None
            elif backward >= 0 and (
                keys[backward] == key or key - keys[backward] < keys[forward] - key
            ):
                # nearest resolving backward: `backward` is already the last
                # row holding the winning value (exact match included).
                match = rows[backward]
            else:
                # nearest resolving forward (equidistant ties included): among
                # duplicates of the winning value polars takes the last row.
                match = rows[bisect_right(keys, keys[forward]) - 1]
            right_indices.append(match)
        # One output row per left row: left columns pass through untouched, the
        # asof key is coalesced and right `by` columns are dropped (like polars).
        right_names = self._join_output_names(
            other, exclude_right={right_on, *(by_right or ())}, suffix=suffix
        )
        take_right = self._take_nullable if None in right_indices else self._take
        result: DictFrame = dict(self.native)
        for name, output_name in right_names.items():
            result[output_name] = take_right(other.native[name], right_indices)
        return self._with_native(result, validate_column_names=False)

    def lazy(
        self,
        backend: _LazyAllowedImpl | None = None,
        *,
        session: SparkSession | None = None,
    ) -> CompliantLazyFrameAny:
        match backend:
            case None:
                return self
            case Implementation.POLARS:
                from narwhals._polars.dataframe import PolarsLazyFrame

                return PolarsLazyFrame(
                    self.to_polars().lazy(),
                    validate_backend_version=True,
                    version=self._version,
                )
            case Implementation.DUCKDB:
                import duckdb  # ignore-banned-import

                from narwhals._duckdb.dataframe import DuckDBLazyFrame

                _df = self.to_arrow()  # `duckdb.table` reads it from this frame
                return DuckDBLazyFrame(
                    duckdb.table("_df"),
                    validate_backend_version=True,
                    version=self._version,
                )
            case Implementation.DASK:
                import dask.dataframe as dd  # ignore-banned-import

                from narwhals._dask.dataframe import DaskLazyFrame

                return DaskLazyFrame(
                    dd.from_pandas(self.to_pandas()),
                    validate_backend_version=True,
                    version=self._version,
                )
            case Implementation.IBIS:
                import ibis  # ignore-banned-import

                from narwhals._ibis.dataframe import IbisLazyFrame

                return IbisLazyFrame(
                    ibis.memtable(self.to_arrow(), columns=self.columns),
                    validate_backend_version=True,
                    version=self._version,
                )
            case backend if backend.is_spark_like():
                from narwhals._spark_like.dataframe import SparkLikeLazyFrame

                if session is None:
                    msg = "Spark like backends require `session` to be not None."
                    raise ValueError(msg)
                return SparkLikeLazyFrame._from_compliant_dataframe(
                    self, session=session, implementation=backend, version=self._version
                )
            case _:  # pragma: no cover
                msg = f"Unsupported `backend` value: {backend}"
                raise ValueError(msg)

    def collect(
        self, backend: _EagerAllowedImpl | None, **kwargs: Any
    ) -> CompliantDataFrameAny:
        match backend:
            case None:
                return self
            case Implementation.PANDAS:
                from narwhals._pandas_like.dataframe import PandasLikeDataFrame

                return PandasLikeDataFrame(
                    self.to_pandas(),
                    implementation=Implementation.PANDAS,
                    validate_backend_version=True,
                    version=self._version,
                    validate_column_names=False,
                )
            case Implementation.PYARROW:
                from narwhals._arrow.dataframe import ArrowDataFrame

                return ArrowDataFrame(
                    self.to_arrow(),
                    validate_backend_version=True,
                    version=self._version,
                    validate_column_names=False,
                )
            case Implementation.POLARS:
                from narwhals._polars.dataframe import PolarsDataFrame

                return PolarsDataFrame(
                    self.to_polars(), validate_backend_version=True, version=self._version
                )
            case _:  # pragma: no cover
                msg = f"Unsupported `backend` value: {backend}"
                raise ValueError(msg)

    def is_unique(self) -> DictSeries:
        counts = Counter(zip(*self.native.values(), strict=True))
        return DictSeries(
            [counts[row] == 1 for row in zip(*self.native.values(), strict=True)],
            name="",
            version=self._version,
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
        return self._gather_positions(indices)

    def _pivot_resolve(
        self, on: Sequence[str], index: Sequence[str] | None, values: Sequence[str] | None
    ) -> tuple[Sequence[str], Sequence[str]]:
        """Fill in `index`/`values` defaults as the complementary columns."""
        on_set = set(on)
        if index is None:
            excluded = on_set | set(values) if values else on_set
            index = [name for name in self.columns if name not in excluded]
        if values is None:
            excluded = on_set | set(index)
            values = [name for name in self.columns if name not in excluded]
        return index, values

    def _pivot_on_uniques(
        self, on: Sequence[str], *, sort_columns: bool
    ) -> Iterator[Iterable[Any]]:
        """Distinct values of each `on` column; the output columns are their product."""
        for name in on:
            unique = dict.fromkeys(self.native[name])  # dedup, first-appearance order
            if not sort_columns:
                yield unique.keys()
            else:
                nulls = (value for value in unique if value is None)
                yield (*nulls, *sorted(value for value in unique if value is not None))

    @staticmethod
    def _pivot_output_name(
        value: str, on_combo: tuple[Any, ...], *, n_on: int, n_values: int, separator: str
    ) -> str:
        label = on_combo[0] if n_on == 1 else '{"' + '","'.join(on_combo) + '"}'
        return f"{value}{separator}{label}" if n_values > 1 else label

    def _pivot_aggregate(
        self,
        column: Sequence[Any],
        rows: Sequence[int],
        aggregate_function: PivotAgg | None,
    ) -> Any:
        # `rows` is the (non-empty) set of original row indices in this cell; only
        # materialize the values when the aggregation actually needs them.
        if aggregate_function == "len":
            return len(rows)
        if aggregate_function is None:
            return column[rows[0]]  # exactly one element per cell (validated in `pivot`)
        series = DictSeries([column[i] for i in rows], name="", version=self._version)
        return getattr(series, aggregate_function)()

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
        if error := self._check_columns_exist(on):
            raise error
        index, values = self._pivot_resolve(on, index, values)
        if error := self._check_columns_exist([*index, *values]):
            raise error

        index_cols = [self.native[name] for name in index]
        on_cols = [self.native[name] for name in on]
        # Stream one (index-key, on-key) pair per row straight into `cells`.
        # No intermediate per-row key lists. `cells` maps each pair to its original
        # row indices in row order, so `first`/`last` see frame order.
        index_keys: Iterator[tuple[Any, ...]] = (
            zip(*index_cols, strict=True) if index_cols else repeat((), len(self))
        )
        cells: dict[tuple[tuple[Any, ...], tuple[Any, ...]], list[int]] = {}
        for row, cell in enumerate(
            zip(index_keys, zip(*on_cols, strict=True), strict=True)
        ):
            cells.setdefault(cell, []).append(row)
        if aggregate_function is None and any(len(rows) > 1 for rows in cells.values()):
            msg = (
                "Found multiple elements for some combinations of `on` and `index` "
                "values. Please pass an `aggregate_function`."
            )
            raise ValueError(msg)

        # Distinct index rows in first-appearance order, recovered from the cell
        # keys (their insertion order already follows row order).
        index_rows = list(dict.fromkeys(index_key for index_key, _ in cells))
        # Materialized once because it is reused for every `value` column.
        on_combos = list(product(*self._pivot_on_uniques(on, sort_columns=sort_columns)))
        n_on, n_values = len(on), len(values)
        result: DictFrame = {
            name: [row[position] for row in index_rows]
            for position, name in enumerate(index)
        }
        for value in values:
            column = self.native[value]
            for combo in on_combos:
                name = self._pivot_output_name(
                    value, combo, n_on=n_on, n_values=n_values, separator=separator
                )
                result[name] = [
                    self._pivot_aggregate(column, rows, aggregate_function)
                    if (rows := cells.get((index_row, combo)))
                    else None
                    for index_row in index_rows
                ]
        return self._with_native(result)

    # Not implemented (yet): fill in incrementally.
    write_parquet = not_implemented()
