from __future__ import annotations

import warnings
from functools import lru_cache
from itertools import chain
from operator import methodcaller
from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

from narwhals._compliant import EagerGroupBy
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals._pandas_like.utils import (
    is_nullable_dtype_backend,
    native_to_narwhals_dtype,
)
from narwhals._typing_compat import TypeVar
from narwhals._utils import find_stacklevel, generate_temporary_column_name

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    import pandas as pd
    from pandas.api.typing import (
        DataFrameGroupBy as _NativeGroupBy,
        SeriesGroupBy as _SeriesGroupBy,
    )
    from typing_extensions import TypeAlias, Unpack

    from narwhals._compliant.typing import NarwhalsAggregation, ScalarKwargs
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.expr import PandasLikeExpr
    from narwhals._utils import Implementation
    from narwhals.dtypes import DType

    NativeGroupBy: TypeAlias = "_NativeGroupBy[tuple[str, ...], Literal[True]]"
    NativeSeriesGroupBy: TypeAlias = "_SeriesGroupBy[Any, tuple[str, ...]]"

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

_NativeAgg: TypeAlias = "NativeAggregation | Callable[..., Any]"
"""Equivalent to `pd.NamedAgg.aggfunc`."""

_NamedAgg: TypeAlias = "tuple[str, _NativeAgg]"
"""Equivalent to `pd.NamedAgg`."""

IterStrT = TypeVar("IterStrT", bound="Iterable[str]")

NonStrHashable: TypeAlias = Any
"""Because `pandas` allows *"names"* like that 😭"""

T = TypeVar("T")


@lru_cache(maxsize=32)
def _native_agg(
    name: NativeAggregation, impl: Implementation, /, **kwds: Unpack[ScalarKwargs]
) -> _NativeAgg:
    if name == "nunique":
        return _n_unique
    if not kwds or kwds.get("ddof") == 1:
        return name
    if impl.is_modin():  # pragma: no cover
        return lambda x: getattr(x, name)(**kwds)
    return methodcaller(name, **kwds)


def _n_unique(self: NativeSeriesGroupBy) -> pd.Series[Any]:
    """`modin` gets confused by `operator.methodcaller`.

    See https://github.com/narwhals-dev/narwhals/pull/2680#discussion_r2151945563
    """
    return self.nunique(dropna=False)


def _remap_non_str(group_by: PandasLikeGroupBy) -> dict[NonStrHashable, str]:
    """Before aggregating, rename every column that isn't already a `str`.

    - Proxy all incoming expressions through a rename mapper
    - At the end, rename back to the original
    - An empty result follows a no-op path in `with_columns.
    """
    original = group_by._original_columns
    exclude = set(group_by.exclude)
    if remaining := set(original).difference(exclude):
        union = exclude.union(original)
        return {
            name: generate_temporary_column_name(8, union)
            for name in remaining
            if not isinstance(name, str)
        }
    return {}  # pragma: no cover


def collect(iterable: tuple[T, ...] | Iterable[T], /) -> tuple[T, ...]:
    """Collect `iterable` into a `tuple`, *iff* it is not one already.

    Borrowed from [`ExprIR` PR].

    [`ExprIR` PR]: https://github.com/narwhals-dev/narwhals/blob/1de65d2f82ace95a9bc72667067ffdfa9d28be6d/narwhals/_plan/common.py#L396-L398
    """
    return iterable if isinstance(iterable, tuple) else tuple(iterable)


class AggExpr:
    """Wrapper storing the intermediate state per-`PandasLikeExpr`.

    There's a lot of edge cases to handle, so aim to evaluate as little
    as possible - and store anything that's needed twice.

    Warning:
        While a `PandasLikeExpr` can be reused - this wrapper is valid **only**
        in a single `.agg(...)` operation.
    """

    expr: PandasLikeExpr
    output_names: tuple[str, ...]
    aliases: tuple[str, ...]

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
        output_names, aliases = evaluate_output_names_and_aliases(self.expr, df, exclude)
        self.output_names, self.aliases = collect(output_names), collect(aliases)
        return self

    def named_aggs(
        self, group_by: PandasLikeGroupBy, /
    ) -> Iterator[tuple[str, _NamedAgg]]:
        aliases = collect(group_by._aliases_str(self.aliases))
        output_names = collect(group_by._aliases_str(self.output_names))
        native_agg = self.native_agg()
        if self.is_len() and self.is_anonymous():
            yield aliases[0], (group_by._anonymous_column_name, native_agg)
            return
        for output_name, alias in zip(output_names, aliases):
            yield alias, (output_name, native_agg)

    def _cast_coerced(self, group_by: PandasLikeGroupBy, /) -> Iterator[PandasLikeExpr]:
        """Yield post-agg casts to correct for weird pandas behavior.

        See https://github.com/narwhals-dev/narwhals/pull/2680#discussion_r2157251589
        """
        df = group_by.compliant
        if self.is_n_unique() and has_non_int_nullable_dtype(df, self.output_names):
            ns = df.__narwhals_namespace__()
            yield ns.col(*self.aliases).cast(ns._version.dtypes.Int64())

    def is_len(self) -> bool:
        return self.leaf_name == "len"

    def is_n_unique(self) -> bool:
        return self.leaf_name == "n_unique"

    def is_anonymous(self) -> bool:
        return self.expr._depth == 0

    @property
    def implementation(self) -> Implementation:
        return self.expr._implementation

    @property
    def kwargs(self) -> ScalarKwargs:
        return self.expr._scalar_kwargs

    @property
    def leaf_name(self) -> NarwhalsAggregation | Any:
        if name := self._leaf_name:
            return name
        self._leaf_name = PandasLikeGroupBy._leaf_name(self.expr)
        return self._leaf_name

    def native_agg(self) -> _NativeAgg:
        return _native_agg(
            PandasLikeGroupBy._remap_expr_name(self.leaf_name),
            self.implementation,
            **self.kwargs,
        )


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
        "quantile": "quantile",
    }
    _original_columns: tuple[str, ...]
    """Column names *prior* to any aliasing in `ParseKeysGroupBy`."""

    _keys: list[str]
    """Stores the **aliased** version of group keys from `ParseKeysGroupBy`."""

    _output_key_names: list[str]
    """Stores the **original** version of group keys."""

    _remap_non_str_columns: dict[NonStrHashable, str]

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
        frame, self._keys, self._output_key_names = self._parse_keys(df, keys=keys)
        self._exclude: tuple[str, ...] = (*self._keys, *self._output_key_names)
        self._remap_non_str_columns = _remap_non_str(self)
        self._compliant_frame = frame.rename(self._remap_non_str_columns)
        # Drop index to avoid potential collisions:
        # https://github.com/narwhals-dev/narwhals/issues/1907.
        native = self.compliant.native
        if set(native.index.names).intersection(self.compliant.columns):
            native = native.reset_index(drop=True)
        self._grouped: NativeGroupBy = native.groupby(
            self._keys.copy(),
            sort=False,
            as_index=True,
            dropna=drop_null_keys,
            observed=True,
        )

    def agg(self, *exprs: PandasLikeExpr) -> PandasLikeDataFrame:
        all_aggs_are_simple = True
        agg_exprs: list[AggExpr] = []
        for expr in exprs:
            agg_exprs.append(AggExpr(expr).with_expand_names(self))
            if not self._is_simple(expr):
                all_aggs_are_simple = False

        if all_aggs_are_simple:
            result: pd.DataFrame
            if named_aggs := self._named_aggs(agg_exprs):
                result = self._grouped.agg(**named_aggs)  # type: ignore[call-overload]
            else:
                result = self.compliant.__native_namespace__().DataFrame(
                    list(self._grouped.groups), columns=self._keys
                )
        elif self.compliant.native.empty:
            raise empty_results_error()
        else:
            result = self._apply_aggs(exprs)
        return self._select_results(result, agg_exprs)

    def _named_aggs(self, exprs: Iterable[AggExpr], /) -> dict[str, _NamedAgg]:
        """Collect all aggregations into a single mapping."""
        return dict(chain.from_iterable(e.named_aggs(self) for e in exprs))

    @overload
    def _aliases_str(self, names: list[str], /) -> list[str]: ...
    @overload
    def _aliases_str(self, names: IterStrT, /) -> list[str] | IterStrT: ...
    def _aliases_str(self, names: IterStrT, /) -> list[str] | IterStrT:
        """If we started with any non `str` column names, return the proxied `str` aliases for `names`."""
        if remap := self._remap_non_str_columns:
            return [remap.get(name, name) for name in names]
        return names

    @property
    def _anonymous_column_name(self) -> str:
        # `len` doesn't exist yet, so just pick a column to call size on
        return next(
            iter(set(self.compliant.columns).difference(self.exclude)), self._keys[0]
        )

    @property
    def _final_renamer(self) -> dict[str, NonStrHashable]:
        remap = self._remap_non_str_columns
        temps = chain(self._keys, remap.values())
        originals = chain(self._output_key_names, remap)
        return dict(zip(temps, originals))

    def _select_results(
        self, df: pd.DataFrame, /, agg_exprs: Sequence[AggExpr]
    ) -> PandasLikeDataFrame:
        """Responsible for remapping temp column names back to original.

        See `ParseKeysGroupBy`.
        """
        # NOTE: Keep `inplace=True` to avoid making a redundant copy.
        # This may need updating, depending on https://github.com/pandas-dev/pandas/pull/51466/files
        df.reset_index(inplace=True)  # noqa: PD002
        new_names = self._aliases_str(chain.from_iterable(e.aliases for e in agg_exprs))
        return (
            self.compliant._with_native(df, validate_column_names=False)
            .simple_select(*self._keys, *new_names)
            .rename(self._final_renamer)
            .with_columns(*chain.from_iterable(e._cast_coerced(self) for e in agg_exprs))
        )

    def _apply_aggs(self, exprs: Iterable[PandasLikeExpr]) -> pd.DataFrame:
        """Stub issue for `include_groups` [pandas-dev/pandas-stubs#1270].

        - [User guide] mentions `include_groups` 4 times without deprecation.
        - [`DataFrameGroupBy.apply`] doc says the default value of `True` is deprecated since `2.2.0`.
        - `False` is explicitly the only *non-deprecated* option, but entirely omitted since [pandas-dev/pandas-stubs#1268].

        [pandas-dev/pandas-stubs#1270]: https://github.com/pandas-dev/pandas-stubs/issues/1270
        [User guide]: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
        [`DataFrameGroupBy.apply`]: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.DataFrameGroupBy.apply.html
        [pandas-dev/pandas-stubs#1268]: https://github.com/pandas-dev/pandas-stubs/pull/1268
        """
        warn_complex_group_by()
        impl = self.compliant._implementation
        func = self._apply_exprs_function(exprs)
        apply = self._grouped.apply
        if impl.is_pandas() and impl._backend_version() >= (2, 2):
            return apply(func, include_groups=False)  # type: ignore[call-overload]
        else:  # pragma: no cover
            return apply(func)

    def _apply_exprs_function(self, exprs: Iterable[PandasLikeExpr]) -> NativeApply:
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
            if self._remap_non_str_columns:
                for key, group in self._grouped:
                    yield (
                        key,
                        with_native(group)
                        .rename({v: k for k, v in self._remap_non_str_columns.items()})
                        .simple_select(*self._original_columns),
                    )
            else:
                for key, group in self._grouped:
                    yield (key, with_native(group).simple_select(*self._original_columns))


def has_non_int_nullable_dtype(
    frame: PandasLikeDataFrame, subset: Sequence[str], /
) -> bool:
    """Return True if any column in `subset` may get incorrectly coerced after aggregation."""
    return bool(subset and any(_has_non_int_nullable_dtype(frame, subset)))


def _has_non_int_nullable_dtype(
    frame: PandasLikeDataFrame, subset: Sequence[str], /
) -> Iterator[bool]:
    version = frame._version
    impl = frame._implementation
    native_dtypes = frame.native.dtypes
    for col in subset:
        native = native_dtypes[col]
        if str(native) != "object":
            dtype = native_to_narwhals_dtype(native, version, impl)
            yield not (dtype.is_integer()) and (
                is_nullable_dtype_backend(native, impl)
                or _is_old_pandas_float(dtype, impl)
            )


PANDAS_FLOAT_FIXED = (1, 3, 5)
"""Keep increasing until random versions doesn't produce `FAILED tests/frame/group_by_test.py::test_group_by_no_preserve_dtype[pandas-float]`"""


def _is_old_pandas_float(dtype: DType, impl: Implementation) -> bool:
    return (
        dtype.is_float()
        and impl.is_pandas()
        and impl._backend_version() < PANDAS_FLOAT_FIXED
    )


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
