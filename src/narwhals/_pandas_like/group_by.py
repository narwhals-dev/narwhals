from __future__ import annotations

import warnings
from functools import lru_cache
from itertools import chain
from operator import methodcaller
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from narwhals._compliant import EagerGroupBy
from narwhals._exceptions import issue_warning
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals._pandas_like.utils import make_group_by_kwargs
from narwhals._utils import check_column_names_are_unique
from narwhals.dependencies import is_pandas_like_dataframe

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from typing import TypeAlias

    import pandas as pd
    from pandas.api.typing import DataFrameGroupBy as _NativeGroupBy
    from typing_extensions import Unpack

    from narwhals._compliant.typing import NarwhalsAggregation, ScalarKwargs
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.expr import PandasLikeExpr
    from narwhals._pandas_like.series import PandasLikeSeries

    NativeGroupBy: TypeAlias = "_NativeGroupBy[tuple[str, ...], Literal[True]]"

InefficientNativeAggregation: TypeAlias = Literal["cov", "skew"]
NativeAggregation: TypeAlias = Literal[
    "any",
    "all",
    "count",
    "idxmax",
    "idxmin",
    "max",
    "mean",
    "median",
    "min",
    "mode",
    "nth",
    "nunique",
    "prod",
    "quantile",
    "sem",
    "size",
    "std",
    "sum",
    "var",
    InefficientNativeAggregation,
]
"""https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#built-in-aggregation-methods"""

_NativeAgg: TypeAlias = "Callable[[Any], pd.DataFrame | pd.Series[Any]]"
"""Equivalent to a partial method call on `DataFrameGroupBy`."""


NonStrHashable: TypeAlias = Any
"""Because `pandas` allows *"names"* like that 😭"""

_REMAP_ORDERED_INDEX: Mapping[NarwhalsAggregation, Literal[0, -1]] = {
    "first": 0,
    "last": -1,
    "any_value": 0,
}


@lru_cache(maxsize=32)
def _native_agg(name: NativeAggregation, /, **kwds: Unpack[ScalarKwargs]) -> _NativeAgg:
    if name == "nunique":
        return methodcaller(name, dropna=False)
    if name == "quantile":
        assert "quantile" in kwds  # noqa: S101
        assert "interpolation" in kwds  # noqa: S101
        return methodcaller(name, q=kwds["quantile"], interpolation=kwds["interpolation"])
    if not kwds or kwds.get("ddof") == 1:
        return methodcaller(name)
    return methodcaller(name, **kwds)


class AggExpr:
    """Wrapper storing the intermediate state per-`PandasLikeExpr`.

    There's a lot of edge cases to handle, so aim to evaluate as little
    as possible - and store anything that's needed twice.

    Warning:
        While a `PandasLikeExpr` can be reused - this wrapper is valid **only**
        in a single `.agg(...)` operation.
    """

    expr: PandasLikeExpr
    output_names: Sequence[str]
    aliases: Sequence[str]

    def __init__(self, expr: PandasLikeExpr) -> None:
        self.expr = expr
        self.output_names = ()
        self.aliases = ()
        self._leaf_name: NarwhalsAggregation | Any = ""

    def with_expand_names(self, group_by: PandasLikeGroupBy, /) -> AggExpr:
        """**Mutating operation**.

        Stores the results of `evaluate_output_names_and_aliases`.
        """
        df = group_by.compliant
        exclude = group_by.exclude
        self.output_names, self.aliases = evaluate_output_names_and_aliases(
            self.expr, df, exclude
        )
        return self

    def evaluate(self, group: PandasLikeDataFrame, /) -> Sequence[PandasLikeSeries]:
        """Evaluate on a single group, keeping selectors (e.g. `nw.all()`) off its keys."""
        if self.expr._metadata.expansion_kind.is_multi_unnamed():
            group = group.simple_select(*self.output_names)
        return self.expr(group)

    def _getitem_aggs(self, group_by: PandasLikeGroupBy) -> pd.DataFrame | pd.Series[Any]:
        """Evaluate the wrapped expression as a group_by operation."""
        grouped = group_by._grouped
        result: pd.DataFrame | pd.Series[Any]
        names = self.output_names
        if self.is_len() and self.is_top_level_function():
            result = grouped.size()
        elif self.is_len():
            result_single = grouped.size()
            ns = group_by.compliant.__narwhals_namespace__()
            result = ns._concat_by_index(
                [ns.from_native(result_single).alias(name).native for name in names]
            )
        elif self.is_mode():
            compliant = group_by.compliant
            node_kwargs = group_by._kwargs(self.expr)
            if (keep := node_kwargs.get("keep")) != "any":  # pragma: no cover
                msg = (
                    f"`Expr.mode(keep='{keep}')` is not implemented in group by context for "
                    f"backend {compliant._implementation}\n\n"
                    "Hint: Use `nw.col(...).mode(keep='any')` instead."
                )
                raise NotImplementedError(msg)

            cols = list(names)
            native = compliant.native
            keys, kwargs = group_by._keys, group_by._group_by_kwargs

            # Implementation based on the following suggestion:
            # https://github.com/pandas-dev/pandas/issues/19254#issuecomment-778661578
            ns = compliant.__narwhals_namespace__()
            result = ns._concat_by_index(
                [
                    native.groupby([*keys, col], **kwargs)
                    .size()
                    .sort_values(ascending=False)
                    .reset_index(col)
                    .groupby(keys, **kwargs)[col]
                    .head(1)
                    .sort_index()
                    for col in cols
                ]
            )
        elif self.is_last() or self.is_first() or self.is_any_value():
            result = self.native_agg()(grouped[[*group_by._keys, *names]])
            impl = group_by.compliant._implementation
            backend_version = impl._backend_version()
            if impl.is_pandas() and backend_version < (3, 0):  # pragma: no cover
                # NOTE: Keep `inplace=True` to avoid making a redundant copy.
                result.set_index(group_by._keys, inplace=True)  # noqa: PD002
            else:
                result = result.set_index(group_by._keys)
        else:
            select = names[0] if len(names) == 1 else list(names)
            result = self.native_agg()(grouped[select])
        if is_pandas_like_dataframe(result):
            result = cast("pd.DataFrame", result)
            result.columns = list(self.aliases)
        else:
            result = cast("pd.Series[Any]", result)
            result.name = self.aliases[0]
        return result

    def is_len(self) -> bool:
        return self.leaf_name == "len"

    def is_last(self) -> bool:
        return self.leaf_name == "last"

    def is_first(self) -> bool:
        return self.leaf_name == "first"

    def is_mode(self) -> bool:
        return self.leaf_name == "mode"

    def is_any_value(self) -> bool:
        return self.leaf_name == "any_value"

    def is_top_level_function(self) -> bool:
        # e.g. `nw.len()`.
        return len(list(self.expr._metadata.op_nodes_reversed())) == 1

    @property
    def leaf_name(self) -> NarwhalsAggregation | Any:
        if name := self._leaf_name:
            return name
        self._leaf_name = PandasLikeGroupBy._leaf_name(self.expr)
        return self._leaf_name

    def native_agg(self) -> _NativeAgg:
        """Return a partial `DataFrameGroupBy` method, missing only `self`."""
        native_name = PandasLikeGroupBy._remap_expr_name(self.leaf_name)
        last_node = next(self.expr._metadata.op_nodes_reversed())
        if self.leaf_name in _REMAP_ORDERED_INDEX:
            if last_node.kwargs.get("ignore_nulls"):
                msg = (
                    "`Expr.any_value(ignore_nulls=True)` is not supported in a `group_by` "
                    "context for pandas-like backend"
                )
                raise NotImplementedError(msg)
            return methodcaller("nth", n=_REMAP_ORDERED_INDEX[self.leaf_name])
        return _native_agg(native_name, **last_node.kwargs)


class PandasLikeGroupBy(
    EagerGroupBy["PandasLikeDataFrame", "PandasLikeExpr", NativeAggregation]
):
    _REMAP_AGGS: ClassVar[Mapping[NarwhalsAggregation, NativeAggregation]] = {
        "sum": "sum",
        "mean": "mean",
        "median": "median",
        "max": "max",
        "min": "min",
        "mode": "mode",
        "std": "std",
        "var": "var",
        "len": "size",
        "n_unique": "nunique",
        "count": "count",
        "quantile": "quantile",
        "all": "all",
        "any": "any",
        "first": "nth",
        "last": "nth",
        "any_value": "nth",
    }
    _original_columns: tuple[str, ...]
    """Column names *prior* to any aliasing in `ParseKeysGroupBy`."""

    _keys: list[str]
    """Stores the **aliased** version of group keys from `ParseKeysGroupBy`."""

    _output_key_names: list[str]
    """Stores the **original** version of group keys."""

    _group_by_kwargs: Mapping[str, bool]
    """Stores keyword arguments for `DataFrame.groupby` other than `by`."""

    @property
    def exclude(self) -> tuple[str, ...]:
        """Group keys to ignore when expanding multi-output aggregations."""
        return self._exclude

    def __init__(
        self,
        df: PandasLikeDataFrame,
        keys: Sequence[PandasLikeExpr] | Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        self._original_columns = tuple(df.columns)
        self._drop_null_keys = drop_null_keys
        self._compliant_frame, self._keys, self._output_key_names = self._parse_keys(
            df, keys
        )
        self._exclude: tuple[str, ...] = (*self._keys, *self._output_key_names)
        self._group_by_kwargs = make_group_by_kwargs(drop_null_keys=drop_null_keys)

        # Drop index to avoid potential collisions:
        # https://github.com/narwhals-dev/narwhals/issues/1907.
        self._native = self.compliant.native
        if set(self._native.index.names).intersection(self.compliant.columns):
            self._native = self._native.reset_index(drop=True)

    def agg(self, *exprs: PandasLikeExpr) -> PandasLikeDataFrame:
        all_aggs_are_simple = True
        agg_exprs: list[AggExpr] = []
        order_by = ()
        for expr in exprs:
            agg_exprs.append(AggExpr(expr).with_expand_names(self))
            if not self._is_simple(expr):
                all_aggs_are_simple = False
            md = next(expr._metadata.op_nodes_reversed())
            if _current_order_by := md.kwargs.get("order_by", ()):
                if order_by and _current_order_by != order_by:
                    msg = f"Only one `order_by` can be specified in `group_by`. Found both {order_by} and {_current_order_by}."
                    raise NotImplementedError(msg)
                order_by = _current_order_by
        aliases = chain.from_iterable(e.aliases for e in agg_exprs)
        check_column_names_are_unique([*self._output_key_names, *aliases])

        native = self._native
        if order_by:
            native = native.sort_values(list(order_by), na_position="first")
        grouped: NativeGroupBy = native.groupby(
            self._keys.copy(), **self._group_by_kwargs
        )
        self._grouped = grouped

        if all_aggs_are_simple:
            result: pd.DataFrame
            if agg_exprs:
                ns = self.compliant.__narwhals_namespace__()
                result = ns._concat_by_index(self._getitem_aggs(agg_exprs))
            else:
                result = self.compliant.__native_namespace__().DataFrame(
                    list(grouped.groups), columns=self._keys
                )
        elif self.compliant.native.empty:
            raise empty_results_error()
        else:
            result = self._per_group_aggs(native, grouped, agg_exprs)

        impl = self.compliant._implementation
        backend_version = impl._backend_version()
        if impl.is_pandas() and backend_version < (3, 0):  # pragma: no cover
            # NOTE: Keep `inplace=True` to avoid making a redundant copy.
            result.reset_index(inplace=True)  # noqa: PD002
        else:
            result = result.reset_index()

        return self._select_results(result, agg_exprs)

    def _select_results(
        self, df: pd.DataFrame, /, agg_exprs: Sequence[AggExpr]
    ) -> PandasLikeDataFrame:
        """Responsible for remapping temp column names back to original.

        See `ParseKeysGroupBy`.
        """
        new_names = chain.from_iterable(e.aliases for e in agg_exprs)
        return (
            self.compliant._with_native(df, validate_column_names=False)
            .simple_select(*self._keys, *new_names)
            .rename(dict(zip(self._keys, self._output_key_names, strict=False)))
        )

    def _getitem_aggs(
        self, exprs: Iterable[AggExpr], /
    ) -> list[pd.DataFrame | pd.Series[Any]]:
        return [e._getitem_aggs(self) for e in exprs]

    def _per_group_aggs(
        self, native: Any, grouped: NativeGroupBy, agg_exprs: Sequence[AggExpr]
    ) -> pd.DataFrame:
        warn_complex_group_by()
        native_ns = self.compliant.__native_namespace__()
        DataFrame = native_ns.DataFrame
        one_row_index = native_ns.RangeIndex(1)
        impl = self.compliant._implementation
        if not impl.is_pandas() or impl._backend_version() < (2, 0):
            keys_in_index = False
        else:
            # NOTE: Unnamed key copies aren't columns, so `apply` keeps the key columns in
            # each group and puts the keys in the index. With no groups it returns the
            # input frame instead, so that case falls back to iterating.
            unnamed_keys = [native[key].rename(None) for key in self._keys]
            grouped = native.groupby(unnamed_keys, **self._group_by_kwargs)
            keys_in_index = grouped.ngroups > 0

        def aggregate_group(native_group: pd.DataFrame) -> pd.DataFrame:
            group = self.compliant._with_native(native_group)
            columns = [] if keys_in_index else [native_group[key] for key in self._keys]
            for agg_expr in agg_exprs:
                columns.extend(series.native for series in agg_expr.evaluate(group))
            # NOTE: A 1-row frame keeps each column's dtype (a Series would upcast ints),
            # and an empty result can't take the index, so it becomes a null row.
            first_rows = {}
            for column in columns:
                first_row = column.iloc[0:1]
                if len(first_row):
                    first_row.index = one_row_index
                first_rows[column.name] = first_row
            return DataFrame(first_rows, index=one_row_index)

        if keys_in_index:
            result = grouped.apply(aggregate_group).droplevel(-1).rename_axis(self._keys)
        elif impl.is_modin():  # pragma: no cover
            # NOTE: Iterating over Modin's distributed groups is slower and can fail.
            result = grouped.apply(aggregate_group).set_index(self._keys)
        else:
            # NOTE: pandas<2 and cuDF don't reliably put the keys in `apply`'s index.
            # With no groups, an empty frame still gives the output columns.
            frames = [aggregate_group(group) for _, group in iter_native_groups(grouped)]
            frames = frames or [aggregate_group(native.iloc[:0]).iloc[:0]]
            result = native_ns.concat(frames).set_index(self._keys)
        # NOTE: A null aggregation makes its column object dtype. `infer_objects` is
        # missing on cuDF and raises on Modin.
        return result.infer_objects() if impl.is_pandas() else result

    def __iter__(self) -> Iterator[tuple[Any, PandasLikeDataFrame]]:
        grouped = self._native.groupby(self._keys.copy(), **self._group_by_kwargs)
        with_native = self.compliant._with_native
        for key, group in iter_native_groups(grouped):
            yield (key, with_native(group).simple_select(*self._original_columns))


def iter_native_groups(grouped: NativeGroupBy) -> Iterator[tuple[Any, pd.DataFrame]]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*a length 1 tuple will be returned",
            category=FutureWarning,
        )
        yield from grouped


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
    issue_warning(
        "Found complex group-by expression, which can't be expressed efficiently with the "
        "pandas API. If you can, please rewrite your query such that group-by aggregations "
        "are simple (e.g. mean, std, min, max, ...). \n\n"
        "Please see: "
        "https://narwhals-dev.github.io/narwhals/concepts/improve_group_by_operation/",
        UserWarning,
    )
