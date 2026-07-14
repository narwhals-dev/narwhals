from __future__ import annotations

import math
from itertools import chain
from typing import TYPE_CHECKING, Any

from narwhals._compliant import EagerNamespace
from narwhals._expression_parsing import (
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals._utils import Implementation, check_column_names_are_unique
from narwhals_dict.dataframe import DictDataFrame
from narwhals_dict.expr import DictExpr
from narwhals_dict.selectors import DictSelectorNamespace
from narwhals_dict.series import DictSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from narwhals._utils import Version
    from narwhals.typing import CorrelationMethod, IntoDType, NonNestedLiteral
    from narwhals_dict.typing import DictFrame, NativeSeries


class DictNamespace(
    EagerNamespace[DictDataFrame, DictSeries, DictExpr, "DictFrame", "NativeSeries"]  # type: ignore[type-var]
):
    _implementation = Implementation.UNKNOWN

    def __init__(self, *, version: Version) -> None:
        self._version = version

    @property
    def _dataframe(self) -> type[DictDataFrame]:
        return DictDataFrame

    @property
    def _expr(self) -> type[DictExpr]:
        return DictExpr

    @property
    def _series(self) -> type[DictSeries]:
        return DictSeries

    @property
    def selectors(self) -> DictSelectorNamespace:
        return DictSelectorNamespace.from_namespace(self)

    def len(self) -> DictExpr:
        return self._expr(
            lambda df: [DictSeries([len(df)], name="len", version=self._version)],
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            version=self._version,
        )

    def lit(self, value: NonNestedLiteral, dtype: IntoDType | None) -> DictExpr:
        # Tuples are normalized to lists so nested literals round-trip as `List`.
        scalar = list(value) if isinstance(value, tuple) else value

        def func(_df: DictDataFrame) -> list[DictSeries]:
            series = DictSeries([scalar], name="literal", version=self._version)
            return [series.cast(dtype) if dtype is not None else series]

        return self._expr(
            func,
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            version=self._version,
        )

    # Horizontal functions: evaluate all exprs, align lengths, then reduce row-wise.
    def _horizontal(
        self,
        exprs: Sequence[DictExpr],
        reducer: Callable[[tuple[Any, ...]], Any],
        fast_reducer: Callable[[tuple[Any, ...]], Any] | None = None,
    ) -> DictExpr:
        # `fast_reducer` must agree with `reducer` on null-free rows; it is applied
        # via `map` when no input column contains nulls, so C-level builtins
        # (`all`, `sum`, ...) skip the per-row Python frame entirely.
        def func(df: DictDataFrame) -> list[DictSeries]:
            series = list(chain.from_iterable(expr(df) for expr in exprs))
            aligned = self._series._align_full_broadcast(*series)
            columns = [s.native for s in aligned]
            func = (
                fast_reducer
                if fast_reducer is not None
                and not any(None in column for column in columns)
                else reducer
            )
            result = list(map(func, zip(*columns, strict=True)))
            return [aligned[0]._with_native(result)]

        return self._expr._from_callable(
            func=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def all_horizontal(self, *exprs: DictExpr, ignore_nulls: bool) -> DictExpr:
        def reducer(row: tuple[Any, ...]) -> Any:
            has_null = False
            for value in row:
                if value is None:
                    has_null = True
                elif not value:
                    return False
            return None if has_null and not ignore_nulls else True

        return self._horizontal(exprs, reducer, fast_reducer=all)

    def any_horizontal(self, *exprs: DictExpr, ignore_nulls: bool) -> DictExpr:
        def reducer(row: tuple[Any, ...]) -> Any:
            has_null = False
            for value in row:
                if value is None:
                    has_null = True
                elif value:
                    return True
            return None if has_null and not ignore_nulls else False

        return self._horizontal(exprs, reducer, fast_reducer=any)

    def sum_horizontal(self, *exprs: DictExpr) -> DictExpr:
        return self._horizontal(
            exprs,
            lambda row: sum(value for value in row if value is not None),
            fast_reducer=sum,
        )

    def mean_horizontal(self, *exprs: DictExpr) -> DictExpr:
        def reducer(row: tuple[Any, ...]) -> Any:
            values = [value for value in row if value is not None]
            return sum(values) / len(values) if values else None

        return self._horizontal(exprs, reducer)

    def min_horizontal(self, *exprs: DictExpr) -> DictExpr:
        return self._horizontal(
            exprs,
            lambda row: min((value for value in row if value is not None), default=None),
            fast_reducer=min,
        )

    def max_horizontal(self, *exprs: DictExpr) -> DictExpr:
        return self._horizontal(
            exprs,
            lambda row: max((value for value in row if value is not None), default=None),
            fast_reducer=max,
        )

    def coalesce(self, *exprs: DictExpr) -> DictExpr:
        return self._horizontal(
            exprs, lambda row: next((value for value in row if value is not None), None)
        )

    def concat_str(
        self, *exprs: DictExpr, separator: str, ignore_nulls: bool
    ) -> DictExpr:
        def reducer(row: tuple[Any, ...]) -> Any:
            if ignore_nulls:
                values = [value for value in row if value is not None]
            elif any(value is None for value in row):
                return None
            else:
                values = list(row)
            return separator.join(str(value) for value in values)

        return self._horizontal(exprs, reducer)

    # `when`/`then`/`otherwise` chains are desugared into nested `when_then` calls,
    # which `EagerNamespace` implements on top of the `_if_then_else` hook.
    def _if_then_else(
        self,
        when: NativeSeries,
        then: NativeSeries,
        otherwise: NativeSeries | None = None,
    ) -> NativeSeries:
        if otherwise is None:
            return [t if w else None for w, t in zip(when, then, strict=True)]
        return [t if w else o for w, t, o in zip(when, then, otherwise, strict=True)]

    # Concatenation of native frames
    def _concat_horizontal(self, dfs: Sequence[DictFrame | Any], /) -> DictFrame:
        names = list(chain.from_iterable(df.keys() for df in dfs))
        check_column_names_are_unique(names)
        # Match Polars: shorter frames are padded with nulls.
        height = max((len(column) for df in dfs for column in df.values()), default=0)
        return {
            name: list(column) + [None] * (height - len(column))
            if len(column) < height
            else column
            for df in dfs
            for name, column in df.items()
        }

    def _concat_vertical(self, dfs: Sequence[DictFrame], /) -> DictFrame:
        columns_0 = list(dfs[0])
        for i, df in enumerate(dfs[1:], start=1):
            if (columns_current := list(df)) != columns_0:
                msg = (
                    "unable to vstack, column names don't match:\n"
                    f"   - dataframe 0: {columns_0}\n"
                    f"   - dataframe {i}: {columns_current}\n"
                )
                raise TypeError(msg)
        return {
            name: list(chain.from_iterable(df[name] for df in dfs)) for name in columns_0
        }

    def _concat_diagonal(self, dfs: Sequence[DictFrame], /) -> DictFrame:
        names = list(dict.fromkeys(chain.from_iterable(df.keys() for df in dfs)))
        lengths = [len(next(iter(df.values()), [])) for df in dfs]
        return {
            name: list(
                chain.from_iterable(
                    df.get(name, [None] * length)
                    for df, length in zip(dfs, lengths, strict=True)
                )
            )
            for name in names
        }

    @staticmethod
    def _pairwise_not_null(a: DictSeries, b: DictSeries) -> tuple[list[Any], list[Any]]:
        pairs = [
            (x, y)
            for x, y in zip(a.native, b.native, strict=True)
            if x is not None and y is not None
        ]
        return [x for x, _ in pairs], [y for _, y in pairs]

    def _binary_scalar_expr(
        self, a: DictExpr, b: DictExpr, func: Callable[[DictSeries, DictSeries], Any]
    ) -> DictExpr:
        def call(df: DictDataFrame) -> list[DictSeries]:
            a_series = df._evaluate_single_output_expr(a)
            b_series = df._evaluate_single_output_expr(b)
            return [
                DictSeries(
                    [func(a_series, b_series)], name=a_series.name, version=self._version
                )
            ]

        return self._expr._from_callable(
            func=call,
            evaluate_output_names=combine_evaluate_output_names(a, b),
            alias_output_names=combine_alias_output_names(a, b),
            context=self,
        )

    def corr(self, a: DictExpr, b: DictExpr, *, method: CorrelationMethod) -> DictExpr:
        if method != "pearson":
            msg = (
                f"`corr` with `method={method!r}` is not supported for the dict backend."
            )
            raise NotImplementedError(msg)

        def pearson(a_series: DictSeries, b_series: DictSeries) -> float | None:
            xs, ys = self._pairwise_not_null(a_series, b_series)
            n = len(xs)
            if n < 2:
                return None
            mean_x, mean_y = sum(xs) / n, sum(ys) / n
            cov = var_x = var_y = 0.0
            for x, y in zip(xs, ys, strict=True):
                dx, dy = x - mean_x, y - mean_y
                cov += dx * dy
                var_x += dx * dx
                var_y += dy * dy
            if var_x == 0 or var_y == 0:
                return None
            return cov / math.sqrt(var_x * var_y)

        return self._binary_scalar_expr(a, b, pearson)

    def cov(self, a: DictExpr, b: DictExpr, *, ddof: int) -> DictExpr:
        def covariance(a_series: DictSeries, b_series: DictSeries) -> float | None:
            xs, ys = self._pairwise_not_null(a_series, b_series)
            n = len(xs)
            if ddof == 0 and n == 1:
                return 0.0
            if n - ddof <= 0:
                return None
            mean_x, mean_y = sum(xs) / n, sum(ys) / n
            total = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
            return total / (n - ddof)

        return self._binary_scalar_expr(a, b, covariance)

    def list(self, *exprs: DictExpr) -> DictExpr:
        def func(df: DictDataFrame) -> list[DictSeries]:
            series = list(chain.from_iterable(expr(df) for expr in exprs))
            aligned = self._series._align_full_broadcast(*series)
            columns = [s.native for s in aligned]
            result = [list(row) for row in zip(*columns, strict=True)]
            return [aligned[0]._with_native(result)]

        return self._expr._from_callable(
            func=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def struct(self, *exprs: DictExpr) -> DictExpr:
        def func(df: DictDataFrame) -> list[DictSeries]:
            series = list(chain.from_iterable(expr(df) for expr in exprs))
            aligned = self._series._align_full_broadcast(*series)
            field_names = [s.name for s in aligned]
            columns = [s.native for s in aligned]
            result = [
                dict(zip(field_names, row, strict=True))
                for row in zip(*columns, strict=True)
            ]
            return [aligned[0]._with_native(result)]

        return self._expr._from_callable(
            func=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )
