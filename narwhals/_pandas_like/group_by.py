from __future__ import annotations

import collections
import warnings
from typing import TYPE_CHECKING, Any, ClassVar

from narwhals._compliant import EagerGroupBy
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals._pandas_like.utils import select_columns_by_name
from narwhals._utils import find_stacklevel

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    import pandas as pd
    from typing_extensions import TypeAlias

    from narwhals._compliant.group_by import NarwhalsAggregation
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.expr import PandasLikeExpr

NativeApply: TypeAlias = "Callable[[pd.DataFrame], pd.Series[Any]]"


class PandasLikeGroupBy(EagerGroupBy["PandasLikeDataFrame", "PandasLikeExpr", str]):
    _REMAP_AGGS: ClassVar[Mapping[NarwhalsAggregation, Any]] = {
        "sum": "sum",
        "mean": "mean",
        "median": "median",
        "max": "max",
        "min": "min",
        "std": "std",
        "var": "var",
        "len": "size",
        "n_unique": "nunique",
        "count": "count",
    }

    def __init__(
        self,
        df: PandasLikeDataFrame,
        keys: Sequence[PandasLikeExpr] | Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        self._df = df
        self._drop_null_keys = drop_null_keys
        self._compliant_frame, self._keys, self._output_key_names = self._parse_keys(
            df, keys=keys
        )
        # Drop index to avoid potential collisions:
        # https://github.com/narwhals-dev/narwhals/issues/1907.
        if set(self.compliant.native.index.names).intersection(self.compliant.columns):
            native_frame = self.compliant.native.reset_index(drop=True)
        else:
            native_frame = self.compliant.native

        self._grouped = native_frame.groupby(
            list(self._keys),
            sort=False,
            as_index=True,
            dropna=drop_null_keys,
            observed=True,
        )

    # NOTE: Still have *quite* a bit of work to do here!
    # -------------------------------------------------------
    # NOTE: `C901`      Too complex                 (25 > 10)
    # NOTE: `PLR0912`   Too many branches           (28 > 12)
    # NOTE: `PLR0914`   Too many local variables    (27 > 15)
    # NOTE: `PLR0915`   Too many statements         (83 > 50)
    def agg(self, *exprs: PandasLikeExpr) -> PandasLikeDataFrame:  # noqa: C901, PLR0912, PLR0914, PLR0915
        new_names: list[str] = self._keys.copy()

        all_aggs_are_simple = True
        exclude = (*self._keys, *self._output_key_names)
        for expr in exprs:
            _, aliases = evaluate_output_names_and_aliases(expr, self.compliant, exclude)
            new_names.extend(aliases)
            if not self._is_simple(expr):
                all_aggs_are_simple = False

        # dict of {output_name: root_name} that we count n_unique on
        # We need to do this separately from the rest so that we
        # can pass the `dropna` kwargs.
        nunique_aggs: dict[str, str] = {}
        simple_aggs: dict[str, list[str]] = collections.defaultdict(list)
        simple_aggs_functions: set[str] = set()

        # ddof to (output_names, aliases) mapping
        std_aggs: dict[int, tuple[list[str], list[str]]] = collections.defaultdict(
            lambda: ([], [])
        )
        var_aggs: dict[int, tuple[list[str], list[str]]] = collections.defaultdict(
            lambda: ([], [])
        )

        expected_old_names: list[str] = []
        simple_agg_new_names: list[str] = []

        if all_aggs_are_simple:  # noqa: PLR1702
            for expr in exprs:
                output_names, aliases = evaluate_output_names_and_aliases(
                    expr, self.compliant, exclude
                )
                if expr._depth == 0:
                    # e.g. `agg(nw.len())`
                    function_name = self._remap_expr_name(expr._function_name)
                    simple_aggs_functions.add(function_name)

                    for alias in aliases:
                        expected_old_names.append(f"{self._keys[0]}_{function_name}")
                        simple_aggs[self._keys[0]].append(function_name)
                        simple_agg_new_names.append(alias)
                    continue

                # e.g. `agg(nw.mean('a'))`
                function_name = self._remap_expr_name(self._leaf_name(expr))
                is_n_unique = function_name == "nunique"
                is_std = function_name == "std"
                is_var = function_name == "var"
                for output_name, alias in zip(output_names, aliases):
                    if is_n_unique:
                        nunique_aggs[alias] = output_name
                    elif is_std and (ddof := expr._scalar_kwargs["ddof"]) != 1:  # pyright: ignore[reportTypedDictNotRequiredAccess]
                        std_aggs[ddof][0].append(output_name)
                        std_aggs[ddof][1].append(alias)
                    elif is_var and (ddof := expr._scalar_kwargs["ddof"]) != 1:  # pyright: ignore[reportTypedDictNotRequiredAccess]
                        var_aggs[ddof][0].append(output_name)
                        var_aggs[ddof][1].append(alias)
                    else:
                        expected_old_names.append(f"{output_name}_{function_name}")
                        simple_aggs[output_name].append(function_name)
                        simple_agg_new_names.append(alias)
                        simple_aggs_functions.add(function_name)

            result_aggs = []

            if simple_aggs:
                # Fast path for single aggregation such as `df.groupby(...).mean()`
                if (
                    len(simple_aggs_functions) == 1
                    and (agg_method := simple_aggs_functions.pop()) != "size"
                    and len(simple_aggs) > 1
                ):
                    result_simple_aggs = getattr(
                        self._grouped[list(simple_aggs.keys())], agg_method
                    )()
                    result_simple_aggs.columns = [
                        f"{a}_{agg_method}" for a in result_simple_aggs.columns
                    ]
                else:
                    result_simple_aggs = self._grouped.agg(simple_aggs)
                    result_simple_aggs.columns = [
                        f"{a}_{b}" for a, b in result_simple_aggs.columns
                    ]
                if not (
                    set(result_simple_aggs.columns) == set(expected_old_names)
                    and len(result_simple_aggs.columns) == len(expected_old_names)
                ):  # pragma: no cover
                    raise safety_assertion_error(
                        expected_old_names, result_simple_aggs.columns
                    )

                # Rename columns, being very careful
                expected_old_names_indices: dict[str, list[int]] = (
                    collections.defaultdict(list)
                )
                for idx, item in enumerate(expected_old_names):
                    expected_old_names_indices[item].append(idx)
                index_map: list[int] = [
                    expected_old_names_indices[item].pop(0)
                    for item in result_simple_aggs.columns
                ]
                result_simple_aggs.columns = [simple_agg_new_names[i] for i in index_map]
                result_aggs.append(result_simple_aggs)

            if nunique_aggs:
                result_nunique_aggs = self._grouped[list(nunique_aggs.values())].nunique(
                    dropna=False
                )
                result_nunique_aggs.columns = list(nunique_aggs.keys())

                result_aggs.append(result_nunique_aggs)

            if std_aggs:
                for ddof, (std_output_names, std_aliases) in std_aggs.items():
                    _aggregation = self._grouped[std_output_names].std(ddof=ddof)
                    # `_aggregation` is a new object so it's OK to operate inplace.
                    _aggregation.columns = std_aliases
                    result_aggs.append(_aggregation)
            if var_aggs:
                for ddof, (var_output_names, var_aliases) in var_aggs.items():
                    _aggregation = self._grouped[var_output_names].var(ddof=ddof)
                    # `_aggregation` is a new object so it's OK to operate inplace.
                    _aggregation.columns = var_aliases
                    result_aggs.append(_aggregation)

            if result_aggs:
                output_names_counter = collections.Counter(
                    c for frame in result_aggs for c in frame
                )
                if any(v > 1 for v in output_names_counter.values()):
                    msg = ""
                    for key, value in output_names_counter.items():
                        if value > 1:
                            msg += f"\n- '{key}' {value} times"
                        else:  # pragma: no cover
                            pass
                    msg = f"Expected unique output names, got:{msg}"
                    raise ValueError(msg)
                namespace = self.compliant.__narwhals_namespace__()
                result = namespace._concat_horizontal(result_aggs)
            else:
                # No aggregation provided
                result = self.compliant.__native_namespace__().DataFrame(
                    list(self._grouped.groups.keys()), columns=self._keys
                )
            return self._select_results(result, new_names)

        if self.compliant.native.empty:
            raise empty_results_error()
        return self._agg_complex(exprs, new_names)

    def _select_results(
        self, df: pd.DataFrame, /, new_names: list[str]
    ) -> PandasLikeDataFrame:
        compliant = self.compliant
        # NOTE: Keep `inplace=True` to avoid making a redundant copy.
        # This may need updating, depending on https://github.com/pandas-dev/pandas/pull/51466/files
        df.reset_index(inplace=True)  # noqa: PD002
        native = select_columns_by_name(
            df, new_names, compliant._backend_version, compliant._implementation
        )
        rename = dict(zip(self._keys, self._output_key_names))
        return compliant._with_native(native).rename(rename)

    def _agg_complex(
        self, exprs: Iterable[PandasLikeExpr], new_names: list[str]
    ) -> PandasLikeDataFrame:
        warn_complex_group_by()
        implementation = self.compliant._implementation
        backend_version = self.compliant._backend_version
        func = self._apply_exprs(exprs)
        if implementation.is_pandas() and backend_version >= (2, 2):
            result = self._grouped.apply(func, include_groups=False)
        else:  # pragma: no cover
            result = self._grouped.apply(func)
        return self._select_results(result, new_names)

    def _apply_exprs(self, exprs: Iterable[PandasLikeExpr]) -> NativeApply:
        ns = self.compliant.__narwhals_namespace__()
        into_series = ns._series.from_iterable

        def fn(df: pd.DataFrame) -> pd.Series[Any]:
            out_group = []
            out_names = []
            for expr in exprs:
                results_keys = expr(self.compliant._with_native(df))
                for keys in results_keys:
                    out_group.append(keys.native.iloc[0])
                    out_names.append(keys.name)
            return into_series(out_group, index=out_names, context=ns).native

        return fn

    def __iter__(self) -> Iterator[tuple[Any, PandasLikeDataFrame]]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*a length 1 tuple will be returned",
                category=FutureWarning,
            )

            for key, group in self._grouped:
                yield (
                    key,
                    self.compliant._with_native(group).simple_select(*self._df.columns),
                )


def safety_assertion_error(
    old_names: Sequence[str], new_names: Sequence[str]
) -> AssertionError:
    msg = (
        f"Safety assertion failed, expected {old_names} "
        f"got {new_names}, "
        "please report a bug at https://github.com/narwhals-dev/narwhals/issues"
    )
    return AssertionError(msg)


def empty_results_error() -> ValueError:
    """Don't even attempt this, it's way too inconsistent across pandas versions."""
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
    return ValueError(msg)


def warn_complex_group_by() -> None:
    warnings.warn(
        "Found complex group-by expression, which can't be expressed efficiently with the "
        "pandas API. If you can, please rewrite your query such that group-by aggregations "
        "are simple (e.g. mean, std, min, max, ...). \n\n"
        "Please see: "
        "https://narwhals-dev.github.io/narwhals/concepts/improve_group_by_operation/",
        UserWarning,
        stacklevel=find_stacklevel(),
    )
