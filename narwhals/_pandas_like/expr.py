from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence

from narwhals._compliant import EagerExpr
from narwhals._expression_parsing import ExprKind
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals._expression_parsing import is_elementary_expression
from narwhals._pandas_like.expr_cat import PandasLikeExprCatNamespace
from narwhals._pandas_like.expr_dt import PandasLikeExprDateTimeNamespace
from narwhals._pandas_like.expr_list import PandasLikeExprListNamespace
from narwhals._pandas_like.expr_name import PandasLikeExprNameNamespace
from narwhals._pandas_like.expr_str import PandasLikeExprStringNamespace
from narwhals._pandas_like.group_by import AGGREGATIONS_TO_PANDAS_EQUIVALENT
from narwhals._pandas_like.series import PandasLikeSeries
from narwhals._pandas_like.utils import rename
from narwhals.dependencies import get_numpy
from narwhals.dependencies import is_numpy_array
from narwhals.exceptions import ColumnNotFoundError

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.namespace import PandasLikeNamespace
    from narwhals.dtypes import DType
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
        function_name: str,
        context: _FullContext,
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
        return self._reuse_series_implementation(
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
        return self._reuse_series_implementation(
            "cum_sum", call_kwargs={"reverse": reverse}
        )

    def shift(self: Self, n: int) -> Self:
        return self._reuse_series_implementation("shift", call_kwargs={"n": n})

    def over(self: Self, partition_by: Sequence[str], kind: ExprKind) -> Self:
        if not is_elementary_expression(self):
            msg = (
                "Only elementary expressions are supported for `.over` in pandas-like backends.\n\n"
                "Please see: "
                "https://narwhals-dev.github.io/narwhals/pandas_like_concepts/improve_group_by_operation/"
            )
            raise NotImplementedError(msg)
        function_name = re.sub(r"(\w+->)", "", self._function_name)
        try:
            pandas_function_name = WINDOW_FUNCTIONS_TO_PANDAS_EQUIVALENT[function_name]
        except KeyError:
            try:
                pandas_function_name = AGGREGATIONS_TO_PANDAS_EQUIVALENT[function_name]
            except KeyError:
                msg = (
                    f"Unsupported function: {function_name} in `over` context.\n\n"
                    f"Supported functions are {', '.join(WINDOW_FUNCTIONS_TO_PANDAS_EQUIVALENT)}\n"
                    f"and {', '.join(AGGREGATIONS_TO_PANDAS_EQUIVALENT)}."
                )
                raise NotImplementedError(msg) from None

        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            output_names, aliases = evaluate_output_names_and_aliases(self, df, [])
            pandas_kwargs = window_kwargs_to_pandas_equivalent(
                function_name, self._call_kwargs
            )

            if function_name == "cum_count":
                plx = self.__narwhals_namespace__()
                df = df.with_columns(~plx.col(*output_names).is_null())
            if function_name.startswith("cum_"):
                reverse = self._call_kwargs["reverse"]
            else:
                assert "reverse" not in self._call_kwargs  # debug assertion  # noqa: S101
                reverse = False
            if reverse:
                # Only select the columns we need to avoid reversing columns
                # unnecessarily
                columns = list(set(partition_by).union(output_names))
                native_frame = df[columns]._native_frame[::-1]
            else:
                native_frame = df._native_frame
            res_native = native_frame.groupby(partition_by)[list(output_names)].transform(
                pandas_function_name, **pandas_kwargs
            )
            result_frame = df._from_native_frame(
                rename(
                    res_native,
                    columns=dict(zip(output_names, aliases)),
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                )
            )
            if reverse:
                return [result_frame[name][::-1] for name in aliases]
            return [result_frame[name] for name in aliases]

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

    def map_batches(
        self: Self,
        function: Callable[[Any], Any],
        return_dtype: DType | type[DType] | None,
    ) -> Self:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            input_series_list = self._call(df)
            output_names = [input_series.name for input_series in input_series_list]
            result = [function(series) for series in input_series_list]
            if is_numpy_array(result[0]) or (
                (np := get_numpy()) is not None and np.isscalar(result[0])
            ):
                result = [
                    df.__narwhals_namespace__()
                    ._create_compliant_series(array)
                    .alias(output_name)
                    for array, output_name in zip(result, output_names)
                ]
            if return_dtype is not None:
                result = [series.cast(return_dtype) for series in result]
            return result

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=self._function_name + "->map_batches",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )

    def cum_count(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series_implementation(
            "cum_count", call_kwargs={"reverse": reverse}
        )

    def cum_min(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series_implementation(
            "cum_min", call_kwargs={"reverse": reverse}
        )

    def cum_max(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series_implementation(
            "cum_max", call_kwargs={"reverse": reverse}
        )

    def cum_prod(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series_implementation(
            "cum_prod", call_kwargs={"reverse": reverse}
        )

    def rank(
        self: Self,
        method: Literal["average", "min", "max", "dense", "ordinal"],
        *,
        descending: bool,
    ) -> Self:
        return self._reuse_series_implementation(
            "rank", call_kwargs={"method": method, "descending": descending}
        )

    @property
    def str(self: Self) -> PandasLikeExprStringNamespace:
        return PandasLikeExprStringNamespace(self)

    @property
    def dt(self: Self) -> PandasLikeExprDateTimeNamespace:
        return PandasLikeExprDateTimeNamespace(self)

    @property
    def cat(self: Self) -> PandasLikeExprCatNamespace:
        return PandasLikeExprCatNamespace(self)

    @property
    def name(self: Self) -> PandasLikeExprNameNamespace:
        return PandasLikeExprNameNamespace(self)

    @property
    def list(self: Self) -> PandasLikeExprListNamespace:
        return PandasLikeExprListNamespace(self)
