from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence

from narwhals._compliant import EagerExpr
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals._pandas_like.group_by import PandasLikeGroupBy
from narwhals._pandas_like.series import PandasLikeSeries
from narwhals.exceptions import ColumnNotFoundError
from narwhals.utils import generate_temporary_column_name

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._expression_parsing import ExprMetadata
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.namespace import PandasLikeNamespace
    from narwhals.utils import Implementation
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

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
}


def window_kwargs_to_pandas_equivalent(
    function_name: str, kwargs: dict[str, object]
) -> dict[str, object]:
    if function_name == "shift":
        pandas_kwargs: dict[str, object] = {"periods": kwargs["n"]}
    elif function_name == "rank":
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
        pandas_kwargs = {
            "min_periods": kwargs["min_samples"],
            "window": kwargs["window_size"],
            "center": kwargs["center"],
        }
    else:  # e.g. std, var
        pandas_kwargs = kwargs
    return pandas_kwargs


class PandasLikeExpr(EagerExpr["PandasLikeDataFrame", PandasLikeSeries]):
    def __init__(
        self: Self,
        call: Callable[[PandasLikeDataFrame], Sequence[PandasLikeSeries]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[PandasLikeDataFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
        call_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._implementation = implementation
        self._backend_version = backend_version
        self._version = version
        self._call_kwargs = call_kwargs or {}
        self._metadata: ExprMetadata | None = None

    def __narwhals_namespace__(self: Self) -> PandasLikeNamespace:
        from narwhals._pandas_like.namespace import PandasLikeNamespace

        return PandasLikeNamespace(
            self._implementation, self._backend_version, version=self._version
        )

    def __narwhals_expr__(self) -> None: ...

    @classmethod
    def from_column_names(
        cls: type[Self],
        evaluate_column_names: Callable[[PandasLikeDataFrame], Sequence[str]],
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
                missing_columns = [
                    x for x in evaluate_column_names(df) if x not in df.columns
                ]
                raise ColumnNotFoundError.from_missing_and_available_column_names(
                    missing_columns=missing_columns,
                    available_columns=df.columns,
                ) from e

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
    def from_column_indices(
        cls: type[Self], *column_indices: int, context: _FullContext
    ) -> Self:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            return [
                PandasLikeSeries(
                    df._native_frame.iloc[:, column_index],
                    implementation=df._implementation,
                    backend_version=df._backend_version,
                    version=df._version,
                )
                for column_index in column_indices
            ]

        return cls(
            func,
            depth=0,
            function_name="nth",
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            implementation=context._implementation,
            backend_version=context._backend_version,
            version=context._version,
        )

    def ewm_mean(
        self: Self,
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

    def cum_sum(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_sum", call_kwargs={"reverse": reverse})

    def shift(self: Self, n: int) -> Self:
        return self._reuse_series("shift", call_kwargs={"n": n})

    def over(
        self: Self,
        partition_by: Sequence[str],
        order_by: Sequence[str] | None,
    ) -> Self:
        if not partition_by:
            # e.g. `nw.col('a').cum_sum().order_by(key)`
            # We can always easily support this as it doesn't require grouping.
            assert order_by is not None  # noqa: S101  # help type-check

            def func(df: PandasLikeDataFrame) -> Sequence[PandasLikeSeries]:
                token = generate_temporary_column_name(8, df.columns)
                df = df.with_row_index(token).sort(
                    *order_by, descending=False, nulls_last=False
                )
                results = self(df.drop([token], strict=True))
                sorting_indices = df[token]
                for s in results:
                    s._scatter_in_place(sorting_indices, s)
                return results
        elif not self._is_elementary():
            msg = (
                "Only elementary expressions are supported for `.over` in pandas-like backends.\n\n"
                "Please see: "
                "https://narwhals-dev.github.io/narwhals/pandas_like_concepts/improve_group_by_operation/"
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
                function_name, self._call_kwargs
            )

            def func(df: PandasLikeDataFrame) -> Sequence[PandasLikeSeries]:
                output_names, aliases = evaluate_output_names_and_aliases(self, df, [])
                if function_name == "cum_count":
                    plx = self.__narwhals_namespace__()
                    df = df.with_columns(~plx.col(*output_names).is_null())

                if function_name.startswith("cum_"):
                    reverse = self._call_kwargs["reverse"]
                else:
                    assert "reverse" not in self._call_kwargs  # noqa: S101
                    reverse = False

                if order_by:
                    columns = list(set(partition_by).union(output_names).union(order_by))
                    token = generate_temporary_column_name(8, columns)
                    df = (
                        df[columns]
                        .with_row_index(token)
                        .sort(*order_by, descending=reverse, nulls_last=reverse)
                    )
                    sorting_indices = df[token]
                elif reverse:
                    columns = list(set(partition_by).union(output_names))
                    df = df[columns][::-1]
                if function_name.startswith("rolling"):
                    rolling = df._native_frame.groupby(partition_by)[
                        list(output_names)
                    ].rolling(**pandas_kwargs)
                    assert pandas_function_name is not None  # help mypy  # noqa: S101
                    if pandas_function_name in {"std", "var"}:
                        res_native = getattr(rolling, pandas_function_name)(
                            ddof=self._call_kwargs["ddof"]
                        )
                    else:
                        res_native = getattr(rolling, pandas_function_name)()
                else:
                    res_native = df._native_frame.groupby(partition_by)[
                        list(output_names)
                    ].transform(pandas_function_name, **pandas_kwargs)
                result_frame = df._with_native(res_native).rename(
                    dict(zip(output_names, aliases))
                )
                results = [result_frame[name] for name in aliases]
                if order_by:
                    for s in results:
                        s._scatter_in_place(sorting_indices, s)
                    return results
                if reverse:
                    return [s[::-1] for s in results]
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

    def cum_count(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_count", call_kwargs={"reverse": reverse})

    def cum_min(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_min", call_kwargs={"reverse": reverse})

    def cum_max(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_max", call_kwargs={"reverse": reverse})

    def cum_prod(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_prod", call_kwargs={"reverse": reverse})

    def rolling_sum(
        self: Self, window_size: int, *, min_samples: int, center: bool
    ) -> Self:
        return self._reuse_series(
            "rolling_sum",
            call_kwargs={
                "window_size": window_size,
                "min_samples": min_samples,
                "center": center,
            },
        )

    def rolling_mean(
        self: Self, window_size: int, *, min_samples: int, center: bool
    ) -> Self:
        return self._reuse_series(
            "rolling_mean",
            call_kwargs={
                "window_size": window_size,
                "min_samples": min_samples,
                "center": center,
            },
        )

    def rolling_std(
        self: Self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._reuse_series(
            "rolling_std",
            call_kwargs={
                "window_size": window_size,
                "min_samples": min_samples,
                "center": center,
                "ddof": ddof,
            },
        )

    def rolling_var(
        self: Self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._reuse_series(
            "rolling_var",
            call_kwargs={
                "window_size": window_size,
                "min_samples": min_samples,
                "center": center,
                "ddof": ddof,
            },
        )

    def rank(
        self: Self,
        method: Literal["average", "min", "max", "dense", "ordinal"],
        *,
        descending: bool,
    ) -> Self:
        return self._reuse_series(
            "rank", call_kwargs={"method": method, "descending": descending}
        )
