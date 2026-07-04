from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any

from narwhals._compliant import EagerNamespace
from narwhals._expression_parsing import (
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals._utils import Implementation, check_column_names_are_unique, not_implemented
from narwhals_dict.dataframe import DictDataFrame
from narwhals_dict.expr import DictExpr
from narwhals_dict.selectors import DictSelectorNamespace
from narwhals_dict.series import DictSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from narwhals._utils import Version
    from narwhals.typing import IntoDType, NonNestedLiteral
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
        def func(_df: DictDataFrame) -> list[DictSeries]:
            series = DictSeries([value], name="literal", version=self._version)
            return [series.cast(dtype) if dtype is not None else series]

        return self._expr(
            func,
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            version=self._version,
        )

    # Horizontal functions: evaluate all exprs, align lengths, then reduce row-wise.
    def _horizontal(
        self, exprs: Sequence[DictExpr], reducer: Callable[[tuple[Any, ...]], Any]
    ) -> DictExpr:
        def func(df: DictDataFrame) -> list[DictSeries]:
            series = list(chain.from_iterable(expr(df) for expr in exprs))
            aligned = self._series._align_full_broadcast(*series)
            result = [
                reducer(row) for row in zip(*(s.native for s in aligned), strict=True)
            ]
            return [aligned[0]._with_native(result)]

        return self._expr._from_callable(
            func=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def all_horizontal(self, *exprs: DictExpr, ignore_nulls: bool) -> DictExpr:
        def reducer(row: tuple[Any, ...]) -> Any:
            values = [value for value in row if value is not None]
            if any(not value for value in values):
                return False
            if not ignore_nulls and len(values) != len(row):
                return None
            return True

        return self._horizontal(exprs, reducer)

    def any_horizontal(self, *exprs: DictExpr, ignore_nulls: bool) -> DictExpr:
        def reducer(row: tuple[Any, ...]) -> Any:
            values = [value for value in row if value is not None]
            if any(values):
                return True
            if not ignore_nulls and len(values) != len(row):
                return None
            return False

        return self._horizontal(exprs, reducer)

    def sum_horizontal(self, *exprs: DictExpr) -> DictExpr:
        return self._horizontal(
            exprs, lambda row: sum(value for value in row if value is not None)
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
        )

    def max_horizontal(self, *exprs: DictExpr) -> DictExpr:
        return self._horizontal(
            exprs,
            lambda row: max((value for value in row if value is not None), default=None),
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
        return {name: column for df in dfs for name, column in df.items()}

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

    corr = not_implemented()
    cov = not_implemented()
    struct = not_implemented()
    when = not_implemented()
