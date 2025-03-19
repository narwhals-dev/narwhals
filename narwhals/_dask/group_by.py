from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Mapping
from typing import Sequence

import dask.dataframe as dd

from narwhals._compliant import CompliantGroupBy
from narwhals._expression_parsing import evaluate_output_names_and_aliases

try:
    import dask.dataframe.dask_expr as dx
except ModuleNotFoundError:  # pragma: no cover
    import dask_expr as dx

if TYPE_CHECKING:
    import pandas as pd
    from pandas.core.groupby import SeriesGroupBy as _PandasSeriesGroupBy
    from typing_extensions import Self
    from typing_extensions import TypeAlias

    from narwhals._dask.dataframe import DaskLazyFrame
    from narwhals._dask.expr import DaskExpr

    PandasSeriesGroupBy: TypeAlias = _PandasSeriesGroupBy[Any, Any]
    _AggFn: TypeAlias = Callable[..., Any]
    Aggregation: TypeAlias = "str | _AggFn"

    from dask_expr._groupby import GroupBy as _DaskGroupBy
else:
    _DaskGroupBy = dx._groupby.GroupBy


def n_unique() -> dd.Aggregation:
    def chunk(s: PandasSeriesGroupBy) -> pd.Series[Any]:
        return s.nunique(dropna=False)

    def agg(s0: PandasSeriesGroupBy) -> pd.Series[Any]:
        return s0.sum()

    return dd.Aggregation(name="nunique", chunk=chunk, agg=agg)


def var(ddof: int) -> _AggFn:
    return partial(_DaskGroupBy.var, ddof=ddof)


def std(ddof: int) -> _AggFn:
    return partial(_DaskGroupBy.std, ddof=ddof)


class DaskLazyGroupBy(CompliantGroupBy["DaskLazyFrame", "DaskExpr"]):
    _NARWHALS_TO_NATIVE_AGGREGATIONS: ClassVar[Mapping[str, Aggregation]] = {
        "sum": "sum",
        "mean": "mean",
        "median": "median",
        "max": "max",
        "min": "min",
        "std": std,
        "var": var,
        "len": "size",
        "n_unique": n_unique,
        "count": "count",
    }

    def __init__(
        self: Self, df: DaskLazyFrame, keys: Sequence[str], /, *, drop_null_keys: bool
    ) -> None:
        self._compliant_frame = df
        self._keys: list[str] = list(keys)
        self._grouped = self.compliant.native.groupby(
            list(self._keys), dropna=drop_null_keys, observed=True
        )

    def agg(self: Self, *exprs: DaskExpr) -> DaskLazyFrame:
        from narwhals._dask.dataframe import DaskLazyFrame

        if not exprs:
            # No aggregation provided
            return self.compliant.simple_select(*self._keys).unique(
                self._keys, keep="any"
            )
        self._ensure_all_simple(exprs)
        # This should be the fastpath, but cuDF is too far behind to use it.
        # - https://github.com/rapidsai/cudf/issues/15118
        # - https://github.com/rapidsai/cudf/issues/15084
        POLARS_TO_DASK_AGGREGATIONS = self._NARWHALS_TO_NATIVE_AGGREGATIONS  # noqa: N806
        simple_aggregations: dict[str, tuple[str, Aggregation]] = {}
        for expr in exprs:
            output_names, aliases = evaluate_output_names_and_aliases(
                expr, self.compliant, self._keys
            )
            if expr._depth == 0:
                # e.g. agg(nw.len()) # noqa: ERA001
                function_name = POLARS_TO_DASK_AGGREGATIONS.get(
                    expr._function_name, expr._function_name
                )
                simple_aggregations.update(
                    dict.fromkeys(aliases, (self._keys[0], function_name))
                )
                continue

            # e.g. agg(nw.mean('a')) # noqa: ERA001
            function_name = re.sub(r"(\w+->)", "", expr._function_name)
            agg_function = POLARS_TO_DASK_AGGREGATIONS.get(function_name, function_name)
            # deal with n_unique case in a "lazy" mode to not depend on dask globally
            agg_function = (
                agg_function(**expr._call_kwargs)
                if callable(agg_function)
                else agg_function
            )
            simple_aggregations.update(
                (alias, (output_name, agg_function))
                for alias, output_name in zip(aliases, output_names)
            )
        return DaskLazyFrame(
            self._grouped.agg(**simple_aggregations).reset_index(),
            backend_version=self.compliant._backend_version,
            version=self.compliant._version,
        )
