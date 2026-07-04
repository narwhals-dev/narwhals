from __future__ import annotations

import operator
from collections.abc import Mapping
from itertools import compress
from typing import TYPE_CHECKING, Any, Literal, overload

from narwhals._compliant import EagerDataFrame
from narwhals._utils import Implementation, check_column_names_are_unique, not_implemented
from narwhals.exceptions import ShapeError
from narwhals_dict.series import DictSeries
from narwhals_dict.utils import is_native_frame

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType

    from typing_extensions import Self, TypeIs

    from narwhals._compliant.typing import CompliantDataFrameAny
    from narwhals._utils import Version, _LimitedContext
    from narwhals.dtypes import DType
    from narwhals.typing import (
        IntoSchema,
        SizedMultiIndexSelector,
        SizedMultiNameSelector,
        _2DArray,
        _SliceIndex,
        _SliceName,
    )
    from narwhals_dict.expr import DictExpr
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

    def sort(self, *by: str, descending: bool | Sequence[bool], nulls_last: bool) -> Self:
        if error := self._check_columns_exist(by):
            raise error
        flags = (
            [descending] * len(by) if isinstance(descending, bool) else list(descending)
        )
        indices = list(range(len(self)))
        # Repeated stable sorts, from the least to the most significant key.
        for name, desc in reversed(list(zip(by, flags, strict=True))):
            column = self.native[name]
            nulls = [i for i in indices if column[i] is None]
            rest = sorted(
                (i for i in indices if column[i] is not None),
                key=lambda i: column[i],
                reverse=desc,
            )
            indices = rest + nulls if nulls_last else nulls + rest
        return self._gather(indices)

    def unique(
        self,
        subset: Sequence[str] | None,
        *,
        keep: Any,
        maintain_order: bool | None = None,
        order_by: Sequence[str] | None = None,
    ) -> Self:
        if order_by:
            msg = "`unique` with `order_by` is not supported for the dict backend."
            raise NotImplementedError(msg)
        if subset is not None and (error := self._check_columns_exist(subset)):
            raise error
        columns = [self.native[name] for name in (subset or self.columns)]
        rows = list(zip(*columns, strict=True))
        if keep in {"any", "first", "last"}:
            seen: dict[Any, int] = {}
            iterable = (
                enumerate(rows) if keep != "last" else reversed(list(enumerate(rows)))
            )
            for index, row in iterable:
                seen.setdefault(row, index)
            indices = sorted(seen.values())
        elif keep == "none":
            from collections import Counter

            counts = Counter(rows)
            indices = [i for i, row in enumerate(rows) if counts[row] == 1]
        else:  # pragma: no cover
            msg = f"Unsupported `keep` strategy: {keep}"
            raise ValueError(msg)
        return self._gather(indices)

    def with_row_index(self, name: str, order_by: Sequence[str] | None) -> Self:
        if order_by:
            msg = (
                "`with_row_index` with `order_by` is not supported for the dict backend."
            )
            raise NotImplementedError(msg)
        return self._with_native({name: list(range(len(self))), **self.native})

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
            return [
                dict(zip(self.columns, row, strict=True))
                for row in zip(*self.native.values(), strict=True)
            ]
        return list(zip(*self.native.values(), strict=True))

    def iter_rows(
        self, *, named: bool, buffer_size: int
    ) -> Iterator[tuple[Any, ...]] | Iterator[dict[str, Any]]:
        # TODO(FBruzzesi): Invert dependencies by implementing `rows` via `iter_rows`
        # In this way we can keep iter_rows lazy and use buffer_size
        if named:
            yield from self.rows(named=True)
        else:
            yield from self.rows(named=False)

    def row(self, index: int) -> tuple[Any, ...]:
        return tuple(column[index] for column in self.native.values())

    def item(self, row: int | None, column: int | str | None) -> Any:
        if row is None and column is None:
            if self.shape != (1, 1):
                msg = f"can only call `.item()` if the dataframe is of shape (1, 1), got: {self.shape}"
                raise ValueError(msg)
            return next(iter(self.native.values()))[0]
        if row is None or column is None:
            msg = "cannot call `.item()` with only one of `row` or `column`"
            raise ValueError(msg)
        name = self.columns[column] if isinstance(column, int) else column
        return self.native[name][row]

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

    # Not implemented (yet): fill in incrementally.
    __array__ = not_implemented()
    estimated_size = not_implemented()
    explode = not_implemented()
    from_arrow = not_implemented()
    group_by = not_implemented()
    is_unique = not_implemented()
    join = not_implemented()
    join_asof = not_implemented()
    pivot = not_implemented()
    sample = not_implemented()
    to_arrow = not_implemented()
    to_numpy = not_implemented()
    to_pandas = not_implemented()
    to_polars = not_implemented()
    unpivot = not_implemented()
    write_csv = not_implemented()
    write_parquet = not_implemented()
