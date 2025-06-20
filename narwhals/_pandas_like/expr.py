from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import EagerExpr
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals._pandas_like.group_by import PandasLikeGroupBy
from narwhals._pandas_like.series import PandasLikeSeries
from narwhals._utils import generate_temporary_column_name

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from narwhals._compliant.typing import AliasNames, EvalNames, EvalSeries, ScalarKwargs
    from narwhals._expression_parsing import ExprMetadata
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.namespace import PandasLikeNamespace
    from narwhals._utils import Implementation, Version, _FullContext
    from narwhals.typing import (
        FillNullStrategy,
        NonNestedLiteral,
        PythonLiteral,
        RankMethod,
    )

WINDOW_FUNCTIONS_TO_PANDAS_EQUIVALENT = {
    "cum_sum": "cumsum",
    "cum_min": "cummin",
    "cum_max": "cummax",
    "cum_prod": "cumprod",
    # Pandas cumcount starts counting from 0 while Polars starts from 1
    # Pandas cumcount counts nulls while Polars does not
    # So, instead of using "cumcount" we use "cumsum" on notna() to get the same result
    "cum_count": "cumsum",
    "rolling_sum": "sum",
    "rolling_mean": "mean",
    "rolling_std": "std",
    "rolling_var": "var",
    "shift": "shift",
    "rank": "rank",
    "diff": "diff",
    "fill_null": "fillna",
}


def window_kwargs_to_pandas_equivalent(
    function_name: str, kwargs: ScalarKwargs
) -> dict[str, PythonLiteral]:
    if function_name == "shift":
        assert "n" in kwargs  # noqa: S101
        pandas_kwargs: dict[str, PythonLiteral] = {"periods": kwargs["n"]}
    elif function_name == "rank":
        assert "method" in kwargs  # noqa: S101
        assert "descending" in kwargs  # noqa: S101
        _method = kwargs["method"]
        pandas_kwargs = {
            "method": "first" if _method == "ordinal" else _method,
            "ascending": not kwargs["descending"],
            "na_option": "keep",
            "pct": False,
        }
    elif function_name.startswith("cum_"):  # Cumulative operation
        pandas_kwargs = {"skipna": True}
    elif function_name.startswith("rolling_"):  # Rolling operation
        assert "min_samples" in kwargs  # noqa: S101
        assert "window_size" in kwargs  # noqa: S101
        assert "center" in kwargs  # noqa: S101
        pandas_kwargs = {
            "min_periods": kwargs["min_samples"],
            "window": kwargs["window_size"],
            "center": kwargs["center"],
        }
    elif function_name in {"std", "var"}:
        assert "ddof" in kwargs  # noqa: S101
        pandas_kwargs = {"ddof": kwargs["ddof"]}
    elif function_name == "fill_null":
        assert "strategy" in kwargs  # noqa: S101
        assert "limit" in kwargs  # noqa: S101
        pandas_kwargs = {"strategy": kwargs["strategy"], "limit": kwargs["limit"]}
    else:  # sum, len, ...
        pandas_kwargs = {}
    return pandas_kwargs


class PandasLikeExpr(EagerExpr["PandasLikeDataFrame", PandasLikeSeries]):
    def __init__(
        self,
        call: EvalSeries[PandasLikeDataFrame, PandasLikeSeries],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: EvalNames[PandasLikeDataFrame],
        alias_output_names: AliasNames | None,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
        scalar_kwargs: ScalarKwargs | None = None,
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._implementation = implementation
        self._backend_version = backend_version
        self._version = version
        self._scalar_kwargs = scalar_kwargs or {}
        self._metadata: ExprMetadata | None = None

    def __narwhals_namespace__(self) -> PandasLikeNamespace:
        from narwhals._pandas_like.namespace import PandasLikeNamespace

        return PandasLikeNamespace(
            self._implementation, self._backend_version, version=self._version
        )

    def __narwhals_expr__(self) -> None: ...

    @classmethod
    def from_column_names(
        cls: type[Self],
        evaluate_column_names: EvalNames[PandasLikeDataFrame],
        /,
        *,
        context: _FullContext,
        function_name: str = "",
    ) -> Self:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            try:
                return [
                    PandasLikeSeries(
                        df._native_frame[column_name],
                        implementation=df._implementation,
                        backend_version=df._backend_version,
                        version=df._version,
                    )
                    for column_name in evaluate_column_names(df)
                ]
            except KeyError as e:
                if error := df._check_columns_exist(evaluate_column_names(df)):
                    raise error from e
                raise

        return cls(
            func,
            depth=0,
            function_name=function_name,
            evaluate_output_names=evaluate_column_names,
            alias_output_names=None,
            implementation=context._implementation,
            backend_version=context._backend_version,
            version=context._version,
        )

    @classmethod
    def from_column_indices(cls, *column_indices: int, context: _FullContext) -> Self:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            native = df.native
            return [
                PandasLikeSeries.from_native(native.iloc[:, i], context=df)
                for i in column_indices
            ]

        return cls(
            func,
            depth=0,
            function_name="nth",
            evaluate_output_names=cls._eval_names_indices(column_indices),
            alias_output_names=None,
            implementation=context._implementation,
            backend_version=context._backend_version,
            version=context._version,
        )

    def ewm_mean(
        self,
        *,
        com: float | None,
        span: float | None,
        half_life: float | None,
        alpha: float | None,
        adjust: bool,
        min_samples: int,
        ignore_nulls: bool,
    ) -> Self:
        return self._reuse_series(
            "ewm_mean",
            com=com,
            span=span,
            half_life=half_life,
            alpha=alpha,
            adjust=adjust,
            min_samples=min_samples,
            ignore_nulls=ignore_nulls,
        )

    def cum_sum(self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_sum", scalar_kwargs={"reverse": reverse})

    def shift(self, n: int) -> Self:
        return self._reuse_series("shift", scalar_kwargs={"n": n})

    def over(  # noqa: C901, PLR0915
        self, partition_by: Sequence[str], order_by: Sequence[str]
    ) -> Self:
        if not partition_by:
            # e.g. `nw.col('a').cum_sum().order_by(key)`
            # We can always easily support this as it doesn't require grouping.
            assert order_by  # noqa: S101

            def func(df: PandasLikeDataFrame) -> Sequence[PandasLikeSeries]:
                token = generate_temporary_column_name(8, df.columns)
                df = df.with_row_index(token).sort(
                    *order_by, descending=False, nulls_last=False
                )
                results = self(df.drop([token], strict=True))
                sorting_indices = df.get_column(token)
                for s in results:
                    s._scatter_in_place(sorting_indices, s)
                return results
        elif not self._is_elementary():
            msg = (
                "Only elementary expressions are supported for `.over` in pandas-like backends.\n\n"
                "Please see: "
                "https://narwhals-dev.github.io/narwhals/concepts/improve_group_by_operation/"
            )
            raise NotImplementedError(msg)
        else:
            function_name = PandasLikeGroupBy._leaf_name(self)
            pandas_function_name = WINDOW_FUNCTIONS_TO_PANDAS_EQUIVALENT.get(
                function_name, PandasLikeGroupBy._REMAP_AGGS.get(function_name)
            )
            if pandas_function_name is None:
                msg = (
                    f"Unsupported function: {function_name} in `over` context.\n\n"
                    f"Supported functions are {', '.join(WINDOW_FUNCTIONS_TO_PANDAS_EQUIVALENT)}\n"
                    f"and {', '.join(PandasLikeGroupBy._REMAP_AGGS)}."
                )
                raise NotImplementedError(msg)
            pandas_kwargs = window_kwargs_to_pandas_equivalent(
                function_name, self._scalar_kwargs
            )

            def func(df: PandasLikeDataFrame) -> Sequence[PandasLikeSeries]:  # noqa: C901, PLR0912
                output_names, aliases = evaluate_output_names_and_aliases(self, df, [])
                if function_name == "cum_count":
                    plx = self.__narwhals_namespace__()
                    df = df.with_columns(~plx.col(*output_names).is_null())

                if function_name.startswith("cum_"):
                    assert "reverse" in self._scalar_kwargs  # noqa: S101
                    reverse = self._scalar_kwargs["reverse"]
                else:
                    assert "reverse" not in self._scalar_kwargs  # noqa: S101
                    reverse = False

                if order_by:
                    columns = list(set(partition_by).union(output_names).union(order_by))
                    token = generate_temporary_column_name(8, columns)
                    df = (
                        df.simple_select(*columns)
                        .with_row_index(token)
                        .sort(*order_by, descending=reverse, nulls_last=reverse)
                    )
                    sorting_indices = df.get_column(token)
                elif reverse:
                    columns = list(set(partition_by).union(output_names))
                    df = df.simple_select(*columns)._gather_slice(slice(None, None, -1))
                grouped = df._native_frame.groupby(partition_by)
                if function_name.startswith("rolling"):
                    rolling = grouped[list(output_names)].rolling(**pandas_kwargs)
                    assert pandas_function_name is not None  # help mypy  # noqa: S101
                    if pandas_function_name in {"std", "var"}:
                        assert "ddof" in self._scalar_kwargs  # noqa: S101
                        res_native = getattr(rolling, pandas_function_name)(
                            ddof=self._scalar_kwargs["ddof"]
                        )
                    else:
                        res_native = getattr(rolling, pandas_function_name)()
                elif function_name == "fill_null":
                    assert "strategy" in self._scalar_kwargs  # noqa: S101
                    assert "limit" in self._scalar_kwargs  # noqa: S101
                    df_grouped = grouped[list(output_names)]
                    if self._scalar_kwargs["strategy"] == "forward":
                        res_native = df_grouped.ffill(limit=self._scalar_kwargs["limit"])
                    elif self._scalar_kwargs["strategy"] == "backward":
                        res_native = df_grouped.bfill(limit=self._scalar_kwargs["limit"])
                    else:  # pragma: no cover
                        # This is deprecated in pandas. Indeed, `nw.col('a').fill_null(3).over('b')`
                        # does not seem very useful, and DuckDB doesn't support it either.
                        msg = "`fill_null` with `over` without `strategy` specified is not supported."
                        raise NotImplementedError(msg)
                elif function_name == "len":
                    if len(output_names) != 1:  # pragma: no cover
                        msg = "Safety check failed, please report a bug."
                        raise AssertionError(msg)
                    res_native = grouped.transform("size").to_frame(aliases[0])
                else:
                    res_native = grouped[list(output_names)].transform(
                        pandas_function_name, **pandas_kwargs
                    )
                result_frame = df._with_native(res_native).rename(
                    dict(zip(output_names, aliases))
                )
                results = [result_frame.get_column(name) for name in aliases]
                if order_by:
                    for s in results:
                        s._scatter_in_place(sorting_indices, s)
                    return results
                if reverse:
                    return [s._gather_slice(slice(None, None, -1)) for s in results]
                return results

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=self._function_name + "->over",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )

    def cum_count(self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_count", scalar_kwargs={"reverse": reverse})

    def cum_min(self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_min", scalar_kwargs={"reverse": reverse})

    def cum_max(self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_max", scalar_kwargs={"reverse": reverse})

    def cum_prod(self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_prod", scalar_kwargs={"reverse": reverse})

    def fill_null(
        self,
        value: Self | NonNestedLiteral,
        strategy: FillNullStrategy | None,
        limit: int | None,
    ) -> Self:
        return self._reuse_series(
            "fill_null", scalar_kwargs={"strategy": strategy, "limit": limit}, value=value
        )

    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._reuse_series(
            "rolling_sum",
            scalar_kwargs={
                "window_size": window_size,
                "min_samples": min_samples,
                "center": center,
            },
        )

    def rolling_mean(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._reuse_series(
            "rolling_mean",
            scalar_kwargs={
                "window_size": window_size,
                "min_samples": min_samples,
                "center": center,
            },
        )

    def rolling_std(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._reuse_series(
            "rolling_std",
            scalar_kwargs={
                "window_size": window_size,
                "min_samples": min_samples,
                "center": center,
                "ddof": ddof,
            },
        )

    def rolling_var(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._reuse_series(
            "rolling_var",
            scalar_kwargs={
                "window_size": window_size,
                "min_samples": min_samples,
                "center": center,
                "ddof": ddof,
            },
        )

    def rank(self, method: RankMethod, *, descending: bool) -> Self:
        return self._reuse_series(
            "rank", scalar_kwargs={"method": method, "descending": descending}
        )

    def log(self, base: float) -> Self:
        return self._reuse_series("log", base=base)

    def exp(self) -> Self:
        return self._reuse_series("exp")

    def sqrt(self) -> Self:
        return self._reuse_series("sqrt")
