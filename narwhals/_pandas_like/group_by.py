from __future__ import annotations

import warnings
from functools import lru_cache
from itertools import chain
from operator import methodcaller
from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

from narwhals._compliant import EagerGroupBy
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals._pandas_like.utils import select_columns_by_name
from narwhals._typing_compat import TypeVar
from narwhals._utils import find_stacklevel, generate_temporary_column_name

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    import pandas as pd
    from pandas.api.typing import DataFrameGroupBy as _NativeGroupBy
    from typing_extensions import TypeAlias, Unpack

    from narwhals._compliant.group_by import NarwhalsAggregation
    from narwhals._compliant.typing import ScalarKwargs
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.expr import PandasLikeExpr

    NativeGroupBy: TypeAlias = "_NativeGroupBy[tuple[str, ...], Literal[True]]"

NativeApply: TypeAlias = "Callable[[pd.DataFrame], pd.Series[Any]]"
InefficientNativeAggregation: TypeAlias = Literal["cov", "skew"]
NativeAggregation: TypeAlias = Literal[
    "any",
    "all",
    "count",
    "first",
    "idxmax",
    "idxmin",
    "last",
    "max",
    "mean",
    "median",
    "min",
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

_AggFunc: TypeAlias = "NativeAggregation | Callable[..., Any]"
"""Equivalent to `pd.NamedAgg.aggfunc`."""

_NamedAgg: TypeAlias = "tuple[str, _AggFunc]"
"""Equivalent to `pd.NamedAgg`."""

SeqStrT = TypeVar("SeqStrT", bound="Sequence[str]", default="list[str]")

NonStrHashable: TypeAlias = Any
"""Because `pandas` allows *"names"* like that 😭"""


@lru_cache(maxsize=32)
def _agg_func(
    name: NativeAggregation, /, **kwds: Unpack[ScalarKwargs]
) -> _AggFunc:  # pragma: no cover
    if name == "nunique":
        return methodcaller(name, dropna=False)
    if not kwds or kwds.get("ddof") == 1:
        return name
    return methodcaller(name, **kwds)


# PLAN
# ----
# - Before aggregating, rename every column that isn't already a `str`
# - Proxy all incoming expressions through a rename mapper
# - At the end, rename back to the original
def _remap_non_str(
    original: Sequence[Any], exclude: Iterable[Any]
) -> dict[NonStrHashable, str]:
    """An empty result follows a no-op path in `with_columns."""
    exclude = set(exclude)
    if remaining := set(original).difference(exclude):
        union = exclude.union(original)
        return {
            name: generate_temporary_column_name(8, union)
            for name in remaining
            if not isinstance(name, str)
        }
    return {}


class PandasLikeGroupBy(
    EagerGroupBy["PandasLikeDataFrame", "PandasLikeExpr", NativeAggregation]
):
    _REMAP_AGGS: ClassVar[Mapping[NarwhalsAggregation, NativeAggregation]] = {
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
    _original_columns: tuple[str, ...]
    """Column names *prior* to any aliasing in `ParseKeysGroupBy`."""

    _keys: list[str]
    """Stores the **aliased** version of group keys from `ParseKeysGroupBy`."""

    _output_key_names: list[str]
    """Stores the **original** version of group keys."""

    _remap_non_str_columns: dict[NonStrHashable, str]

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
        ns = df.__narwhals_namespace__()
        frame, self._keys, self._output_key_names = self._parse_keys(df, keys=keys)
        self._remap_non_str_columns = _remap_non_str(
            self._original_columns, (*self._keys, *self._output_key_names)
        )
        self._compliant_frame = frame.with_columns(
            *(ns.col(old).alias(new) for old, new in self._remap_non_str_columns.items())
        )
        # Drop index to avoid potential collisions:
        # https://github.com/narwhals-dev/narwhals/issues/1907.
        if set(self.compliant.native.index.names).intersection(self.compliant.columns):
            native_frame = self.compliant.native.reset_index(drop=True)
        else:
            native_frame = self.compliant.native

        self._grouped: NativeGroupBy = native_frame.groupby(
            list(self._keys),
            sort=False,
            as_index=True,
            dropna=drop_null_keys,
            observed=True,
        )

    def agg(self, *exprs: PandasLikeExpr) -> PandasLikeDataFrame:
        new_names: list[str] = self._keys.copy()
        all_aggs_are_simple = True
        exclude = (*self._keys, *self._output_key_names)
        for expr in exprs:
            _, aliases = evaluate_output_names_and_aliases(expr, self.compliant, exclude)
            new_names.extend(aliases)
            if not self._is_simple(expr):
                all_aggs_are_simple = False

        if any(not isinstance(k, str) for k in new_names):
            new_names = self._remap_aliases(new_names)
        if all_aggs_are_simple:
            result: pd.DataFrame
            if named_aggs := self._named_aggs(*exprs, exclude=exclude):
                result = self._grouped.agg(**named_aggs)  # type: ignore[call-overload]
            else:
                result = self.compliant.__native_namespace__().DataFrame(
                    list(self._grouped.groups.keys()), columns=self._keys
                )
            return self._select_results(result, new_names)
        if self.compliant.native.empty:
            raise empty_results_error()
        return self._agg_complex(exprs, new_names)

    @overload
    def _remap_aliases(self, names: list[str], /) -> list[str]: ...
    @overload
    def _remap_aliases(self, names: SeqStrT, /) -> list[str] | SeqStrT: ...
    def _remap_aliases(self, names: SeqStrT, /) -> list[str] | SeqStrT:
        if remap := self._remap_non_str_columns:
            return [remap.get(name, name) for name in names]
        return names

    def _named_aggs(
        self, *exprs: PandasLikeExpr, exclude: Sequence[str]
    ) -> dict[str, _NamedAgg]:
        """Collect all aggregations into a single mapping."""
        return dict(chain.from_iterable(self._iter_named_aggs(e, exclude) for e in exprs))

    def _iter_named_aggs(
        self, expr: PandasLikeExpr, exclude: Sequence[str]
    ) -> Iterator[tuple[str, _NamedAgg]]:
        output_names, aliases = evaluate_output_names_and_aliases(
            expr, self.compliant, exclude
        )
        aliases = self._remap_aliases(aliases)
        leaf_name = self._leaf_name(expr)
        function_name = self._remap_expr_name(leaf_name)
        aggfunc = _agg_func(function_name, **expr._scalar_kwargs)
        if leaf_name == "len" and expr._depth == 0:
            # `len` doesn't exist yet, so just pick a column to call size on
            first_col = next(iter(set(self.compliant.columns).difference(exclude)))
            yield aliases[0], (first_col, aggfunc)
        else:
            for output_name, alias in zip(output_names, aliases):
                yield alias, (output_name, aggfunc)

    @property
    def _final_renamer(self) -> dict[str, NonStrHashable]:
        remap = self._remap_non_str_columns
        temps = chain(self._keys, remap.values())
        originals = chain(self._output_key_names, remap)
        return dict(zip(temps, originals))

    def _select_results(
        self, df: pd.DataFrame, /, new_names: list[str]
    ) -> PandasLikeDataFrame:
        """Responsible for remapping temp column names back to original.

        See `ParseKeysGroupBy`.
        """
        compliant = self.compliant
        # NOTE: Keep `inplace=True` to avoid making a redundant copy.
        # This may need updating, depending on https://github.com/pandas-dev/pandas/pull/51466/files
        df.reset_index(inplace=True)  # noqa: PD002
        native = select_columns_by_name(
            df, new_names, compliant._backend_version, compliant._implementation
        )
        return compliant._with_native(native).rename(self._final_renamer)

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
            with_native = self.compliant._with_native
            for key, group in self._grouped:
                yield (key, with_native(group).simple_select(*self._original_columns))


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
