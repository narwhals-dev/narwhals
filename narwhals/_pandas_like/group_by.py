from __future__ import annotations

import collections
import warnings
from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterator

from narwhals._expression_parsing import is_simple_aggregation
from narwhals._expression_parsing import parse_into_exprs
from narwhals._pandas_like.utils import native_series_from_iterable
from narwhals.utils import Implementation
from narwhals.utils import remove_prefix

if TYPE_CHECKING:
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.expr import PandasLikeExpr
    from narwhals._pandas_like.typing import IntoPandasLikeExpr

POLARS_TO_PANDAS_AGGREGATIONS = {
    "len": "size",
    "n_unique": "nunique",
}


class PandasLikeGroupBy:
    def __init__(self, df: PandasLikeDataFrame, keys: list[str]) -> None:
        self._df = df
        self._keys = keys
        if (
            self._df._implementation is Implementation.PANDAS
            and self._df._backend_version < (1, 0)
        ):  # pragma: no cover
            if self._df._native_frame.loc[:, self._keys].isna().any().any():
                msg = "Grouping by null values is not supported in pandas < 1.0.0"
                raise NotImplementedError(msg)
            self._grouped = self._df._native_frame.groupby(
                list(self._keys),
                sort=False,
                as_index=True,
            )
        else:
            self._grouped = self._df._native_frame.groupby(
                list(self._keys),
                sort=False,
                as_index=True,
                dropna=False,
            )

    def agg(
        self,
        *aggs: IntoPandasLikeExpr,
        **named_aggs: IntoPandasLikeExpr,
    ) -> PandasLikeDataFrame:
        exprs = parse_into_exprs(
            *aggs,
            namespace=self._df.__narwhals_namespace__(),
            **named_aggs,
        )
        implementation: Implementation = self._df._implementation
        output_names: list[str] = copy(self._keys)
        for expr in exprs:
            if expr._output_names is None:
                msg = (
                    "Anonymous expressions are not supported in group_by.agg.\n"
                    "Instead of `nw.all()`, try using a named expression, such as "
                    "`nw.col('a', 'b')`\n"
                )
                raise ValueError(msg)
            output_names.extend(expr._output_names)

        return agg_pandas(
            self._grouped,
            exprs,
            self._keys,
            output_names,
            self._from_native_frame,
            dataframe_is_empty=self._df._native_frame.empty,
            implementation=implementation,
            backend_version=self._df._backend_version,
        )

    def _from_native_frame(self, df: PandasLikeDataFrame) -> PandasLikeDataFrame:
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        return PandasLikeDataFrame(
            df,
            implementation=self._df._implementation,
            backend_version=self._df._backend_version,
        )

    def __iter__(self) -> Iterator[tuple[Any, PandasLikeDataFrame]]:
        with warnings.catch_warnings():
            # we already use `tupleify` above, so we're already opting in to
            # the new behaviour
            warnings.filterwarnings(
                "ignore",
                message="In a future version of pandas, a length 1 tuple will be returned",
                category=FutureWarning,
            )
            iterator = self._grouped.__iter__()
        yield from ((key, self._from_native_frame(sub_df)) for (key, sub_df) in iterator)


def agg_pandas(  # noqa: PLR0915
    grouped: Any,
    exprs: list[PandasLikeExpr],
    keys: list[str],
    output_names: list[str],
    from_dataframe: Callable[[Any], PandasLikeDataFrame],
    *,
    implementation: Any,
    backend_version: tuple[int, ...],
    dataframe_is_empty: bool,
) -> PandasLikeDataFrame:
    """
    This should be the fastpath, but cuDF is too far behind to use it.

    - https://github.com/rapidsai/cudf/issues/15118
    - https://github.com/rapidsai/cudf/issues/15084
    """
    all_aggs_are_simple = True
    for expr in exprs:
        if not is_simple_aggregation(expr):
            all_aggs_are_simple = False
            break

    # dict of {output_name: root_name} that we count n_unique on
    # We need to do this separately from the rest so that we
    # can pass the `dropna` kwargs.
    nunique_aggs: dict[str, str] = {}

    if all_aggs_are_simple:
        simple_aggregations: dict[str, tuple[str, str]] = {}
        for expr in exprs:
            if expr._depth == 0:
                # e.g. agg(nw.len()) # noqa: ERA001
                if expr._output_names is None:  # pragma: no cover
                    msg = "Safety assertion failed, please report a bug to https://github.com/narwhals-dev/narwhals/issues"
                    raise AssertionError(msg)

                function_name = POLARS_TO_PANDAS_AGGREGATIONS.get(
                    expr._function_name, expr._function_name
                )
                for output_name in expr._output_names:
                    simple_aggregations[output_name] = (keys[0], function_name)
                continue

            # e.g. agg(nw.mean('a')) # noqa: ERA001
            if (
                expr._depth != 1 or expr._root_names is None or expr._output_names is None
            ):  # pragma: no cover
                msg = "Safety assertion failed, please report a bug to https://github.com/narwhals-dev/narwhals/issues"
                raise AssertionError(msg)

            function_name = remove_prefix(expr._function_name, "col->")
            function_name = POLARS_TO_PANDAS_AGGREGATIONS.get(
                function_name, function_name
            )
            for root_name, output_name in zip(expr._root_names, expr._output_names):
                if function_name == "nunique":
                    nunique_aggs[output_name] = root_name
                else:
                    simple_aggregations[output_name] = (root_name, function_name)

        simple_aggs = collections.defaultdict(list)
        name_mapping = {}
        for output_name, named_agg in simple_aggregations.items():
            simple_aggs[named_agg[0]].append(named_agg[1])
            name_mapping[f"{named_agg[0]}_{named_agg[1]}"] = output_name
        if simple_aggs:
            try:
                result_simple_aggs = grouped.agg(simple_aggs)
            except AttributeError as exc:
                msg = "Failed to aggregated - does your aggregation function return a scalar?"
                raise RuntimeError(msg) from exc
            result_simple_aggs.columns = [
                f"{a}_{b}" for a, b in result_simple_aggs.columns
            ]
            result_simple_aggs = result_simple_aggs.rename(
                columns=name_mapping
            ).reset_index()
        if nunique_aggs:
            result_nunique_aggs = grouped[list(nunique_aggs.values())].nunique(
                dropna=False
            )
            result_nunique_aggs.columns = list(nunique_aggs.keys())
            result_nunique_aggs = result_nunique_aggs.reset_index()
        if simple_aggs and nunique_aggs:
            if (
                set(result_simple_aggs.columns)
                .difference(keys)
                .intersection(result_nunique_aggs.columns)
            ):
                msg = (
                    "Got two aggregations with the same output name. Please make sure "
                    "that aggregations have unique output names."
                )
                raise ValueError(msg)
            result_aggs = result_simple_aggs.merge(result_nunique_aggs, on=keys)
        elif nunique_aggs and not simple_aggs:
            result_aggs = result_nunique_aggs
        elif simple_aggs and not nunique_aggs:
            result_aggs = result_simple_aggs
        else:  # pragma: no cover
            msg = "Congrats, you entered unreachable code. Please report a bug to https://github.com/narwhals-dev/narwhals/issues."
            raise RuntimeError(msg)
        return from_dataframe(result_aggs.loc[:, output_names])

    if dataframe_is_empty:
        # Don't even attempt this, it's way too inconsistent across pandas versions.
        msg = (
            "No results for group-by aggregation.\n\n"
            "Hint: you were probably trying to apply a non-elementary aggregation with a "
            "pandas-like API.\n"
            "Please rewrite your query such that group-by aggregations "
            "are elementary. For example, instead of:\n\n"
            "    df.group_by('a').agg(nw.col('b').round(2).mean())\n\n"
            "use:\n\n"
            "    df.with_columns(nw.col('b').round(2)).group_by('a').agg(nw.col('b').mean())\n\n"
        )
        raise ValueError(msg)

    warnings.warn(
        "Found complex group-by expression, which can't be expressed efficiently with the "
        "pandas API. If you can, please rewrite your query such that group-by aggregations "
        "are simple (e.g. mean, std, min, max, ...).",
        UserWarning,
        stacklevel=2,
    )

    def func(df: Any) -> Any:
        out_group = []
        out_names = []
        for expr in exprs:
            results_keys = expr._call(from_dataframe(df))
            for result_keys in results_keys:
                out_group.append(result_keys._native_series.iloc[0])
                out_names.append(result_keys.name)
        return native_series_from_iterable(
            out_group,
            index=out_names,
            name="",
            implementation=implementation,
        )

    if implementation is Implementation.PANDAS and backend_version >= (2, 2):
        result_complex = grouped.apply(func, include_groups=False)
    else:  # pragma: no cover
        result_complex = grouped.apply(func)

    result = result_complex.reset_index()

    return from_dataframe(result.loc[:, output_names])
