from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import cast

from narwhals import dtypes
from narwhals._expression_parsing import parse_into_exprs
from narwhals._pandas_like.dataframe import PandasLikeDataFrame
from narwhals._pandas_like.expr import PandasLikeExpr
from narwhals._pandas_like.selectors import PandasSelectorNamespace
from narwhals._pandas_like.series import PandasLikeSeries
from narwhals._pandas_like.utils import create_native_series
from narwhals._pandas_like.utils import horizontal_concat
from narwhals._pandas_like.utils import vertical_concat

if TYPE_CHECKING:
    from narwhals._pandas_like.typing import IntoPandasLikeExpr
    from narwhals.utils import Implementation


class PandasLikeNamespace:
    Int64 = dtypes.Int64
    Int32 = dtypes.Int32
    Int16 = dtypes.Int16
    Int8 = dtypes.Int8
    UInt64 = dtypes.UInt64
    UInt32 = dtypes.UInt32
    UInt16 = dtypes.UInt16
    UInt8 = dtypes.UInt8
    Float64 = dtypes.Float64
    Float32 = dtypes.Float32
    Boolean = dtypes.Boolean
    Object = dtypes.Object
    Unknown = dtypes.Unknown
    Categorical = dtypes.Categorical
    Enum = dtypes.Enum
    String = dtypes.String
    Datetime = dtypes.Datetime
    Duration = dtypes.Duration
    Date = dtypes.Date

    @property
    def selectors(self) -> PandasSelectorNamespace:
        return PandasSelectorNamespace(
            implementation=self._implementation, backend_version=self._backend_version
        )

    # --- not in spec ---
    def __init__(
        self, implementation: Implementation, backend_version: tuple[int, ...]
    ) -> None:
        self._implementation = implementation
        self._backend_version = backend_version

    def _create_expr_from_callable(
        self,
        func: Callable[[PandasLikeDataFrame], list[PandasLikeSeries]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
    ) -> PandasLikeExpr:
        return PandasLikeExpr(
            func,
            depth=depth,
            function_name=function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    def _create_series_from_scalar(
        self, value: Any, series: PandasLikeSeries
    ) -> PandasLikeSeries:
        return PandasLikeSeries._from_iterable(
            [value],
            name=series._native_series.name,
            index=series._native_series.index[0:1],
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    def _create_expr_from_series(self, series: PandasLikeSeries) -> PandasLikeExpr:
        return PandasLikeExpr(
            lambda _df: [series],
            depth=0,
            function_name="series",
            root_names=None,
            output_names=None,
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    def _create_compliant_series(self, value: Any) -> PandasLikeSeries:
        return create_native_series(
            value,
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    # --- selection ---
    def col(self, *column_names: str) -> PandasLikeExpr:
        return PandasLikeExpr.from_column_names(
            *column_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    def all(self) -> PandasLikeExpr:
        return PandasLikeExpr(
            lambda df: [
                PandasLikeSeries(
                    df._native_frame.loc[:, column_name],
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                )
                for column_name in df.columns
            ],
            depth=0,
            function_name="all",
            root_names=None,
            output_names=None,
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    def lit(self, value: Any, dtype: dtypes.DType | None) -> PandasLikeExpr:
        def _lit_pandas_series(df: PandasLikeDataFrame) -> PandasLikeSeries:
            pandas_series = PandasLikeSeries._from_iterable(
                data=[value],
                name="lit",
                index=df._native_frame.index[0:1],
                implementation=self._implementation,
                backend_version=self._backend_version,
            )
            if dtype:
                return pandas_series.cast(dtype)
            return pandas_series

        return PandasLikeExpr(
            lambda df: [_lit_pandas_series(df)],
            depth=0,
            function_name="lit",
            root_names=None,
            output_names=["lit"],
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    # --- reduction ---
    def sum(self, *column_names: str) -> PandasLikeExpr:
        return PandasLikeExpr.from_column_names(
            *column_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
        ).sum()

    def mean(self, *column_names: str) -> PandasLikeExpr:
        return PandasLikeExpr.from_column_names(
            *column_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
        ).mean()

    def max(self, *column_names: str) -> PandasLikeExpr:
        return PandasLikeExpr.from_column_names(
            *column_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
        ).max()

    def min(self, *column_names: str) -> PandasLikeExpr:
        return PandasLikeExpr.from_column_names(
            *column_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
        ).min()

    def len(self) -> PandasLikeExpr:
        return PandasLikeExpr(
            lambda df: [
                PandasLikeSeries._from_iterable(
                    [len(df._native_frame)],
                    name="len",
                    index=[0],
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                )
            ],
            depth=0,
            function_name="len",
            root_names=None,
            output_names=["len"],
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    # --- horizontal ---
    def sum_horizontal(self, *exprs: IntoPandasLikeExpr) -> PandasLikeExpr:
        return reduce(
            lambda x, y: x + y,
            [expr.fill_null(0) for expr in parse_into_exprs(*exprs, namespace=self)],
        )

    def all_horizontal(self, *exprs: IntoPandasLikeExpr) -> PandasLikeExpr:
        return reduce(lambda x, y: x & y, parse_into_exprs(*exprs, namespace=self))

    def any_horizontal(self, *exprs: IntoPandasLikeExpr) -> PandasLikeExpr:
        return reduce(lambda x, y: x | y, parse_into_exprs(*exprs, namespace=self))

    def mean_horizontal(self, *exprs: IntoPandasLikeExpr) -> PandasLikeExpr:
        pandas_like_exprs = parse_into_exprs(*exprs, namespace=self)
        total = reduce(lambda x, y: x + y, (e.fill_null(0.0) for e in pandas_like_exprs))
        n_non_zero = reduce(
            lambda x, y: x + y, ((1 - e.is_null()) for e in pandas_like_exprs)
        )
        return total / n_non_zero

    def concat(
        self,
        items: Iterable[PandasLikeDataFrame],
        *,
        how: str = "vertical",
    ) -> PandasLikeDataFrame:
        dfs: list[Any] = [item._native_frame for item in items]
        if how == "horizontal":
            return PandasLikeDataFrame(
                horizontal_concat(
                    dfs,
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                ),
                implementation=self._implementation,
                backend_version=self._backend_version,
            )
        if how == "vertical":
            return PandasLikeDataFrame(
                vertical_concat(
                    dfs,
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                ),
                implementation=self._implementation,
                backend_version=self._backend_version,
            )
        raise NotImplementedError

    def when(
        self,
        *predicates: IntoPandasLikeExpr,
    ) -> PandasWhen:
        plx = self.__class__(self._implementation, self._backend_version)
        if predicates:
            condition = plx.all_horizontal(*predicates)
        else:
            msg = "at least one predicate needs to be provided"
            raise TypeError(msg)

        return PandasWhen(condition, self._implementation, self._backend_version)


class PandasWhen:
    def __init__(
        self,
        condition: PandasLikeExpr,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        then_value: Any = None,
        otherwise_value: Any = None,
    ) -> None:
        self._implementation = implementation
        self._backend_version = backend_version
        self._condition = condition
        self._then_value = then_value
        self._otherwise_value = otherwise_value

        self._already_set = self._condition

    def __call__(self, df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
        from narwhals._expression_parsing import parse_into_expr
        from narwhals._pandas_like.namespace import PandasLikeNamespace
        from narwhals._pandas_like.utils import validate_column_comparand

        plx = PandasLikeNamespace(
            implementation=self._implementation, backend_version=self._backend_version
        )

        condition = parse_into_expr(self._condition, namespace=plx)._call(df)[0]  # type: ignore[arg-type]
        try:
            value_series = parse_into_expr(self._then_value, namespace=plx)._call(df)[0]  # type: ignore[arg-type]
        except TypeError:
            # `self._otherwise_value` is a scalar and can't be converted to an expression
            value_series = condition.__class__._from_iterable(  # type: ignore[call-arg]
                [self._then_value] * len(condition),
                name="literal",
                index=condition._native_series.index,
                implementation=self._implementation,
                backend_version=self._backend_version,
            )
        value_series = cast(PandasLikeSeries, value_series)

        value_series_native = value_series._native_series
        condition_native = validate_column_comparand(value_series_native.index, condition)

        if self._otherwise_value is None:
            return [
                value_series._from_native_series(
                    value_series_native.where(condition_native)
                )
            ]
        try:
            otherwise_series = parse_into_expr(
                self._otherwise_value, namespace=plx
            )._call(df)[0]  # type: ignore[arg-type]
        except TypeError:
            # `self._otherwise_value` is a scalar and can't be converted to an expression
            return [
                value_series._from_native_series(
                    value_series_native.where(condition_native, self._otherwise_value)
                )
            ]
        else:
            return [value_series.zip_with(condition, otherwise_series)]

    def then(self, value: PandasLikeExpr | PandasLikeSeries | Any) -> PandasThen:
        self._then_value = value

        return PandasThen(
            self,
            depth=0,
            function_name="whenthen",
            root_names=None,
            output_names=None,
            implementation=self._implementation,
            backend_version=self._backend_version,
        )


class PandasThen(PandasLikeExpr):
    def __init__(
        self,
        call: PandasWhen,
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        implementation: Implementation,
        backend_version: tuple[int, ...],
    ) -> None:
        self._implementation = implementation
        self._backend_version = backend_version

        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names

    def otherwise(self, value: PandasLikeExpr | PandasLikeSeries | Any) -> PandasThen:
        # type ignore because we are setting the `_call` attribute to a
        # callable object of type `PandasWhen`, base class has the attribute as
        # only a `Callable`
        self._call._otherwise_value = value  # type: ignore[attr-defined]
        self._function_name = "whenotherwise"
        return self

    def when(self, *predicates: IntoPandasLikeExpr) -> PandasChainedWhen:
        plx = PandasLikeNamespace(self._implementation, self._backend_version)
        if predicates:
            condition = plx.all_horizontal(*predicates)
        else:
            msg = "at least one predicate needs to be provided"
            raise TypeError(msg)
        return PandasChainedWhen(
            self, condition, self._depth + 1, self._implementation, self._backend_version
        )


class PandasChainedWhen:
    def __init__(
        self,
        above_then: PandasThen | PandasChainedThen,
        condition: PandasLikeExpr,
        depth: int,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        then_value: Any = None,
        otherise_value: Any = None,
    ) -> None:
        self._implementation = implementation
        self._depth = depth
        self._backend_version = backend_version
        self._condition = condition
        self._above_then = above_then
        self._then_value = then_value
        self._otherwise_value = otherise_value

        # TODO @aivanoved: this is way slow as during computation time this takes
        # quadratic time need to improve this to linear time
        self._above_already_set = self._above_then._call._already_set  # type: ignore[attr-defined]
        self._already_set = self._above_already_set | self._condition

    def __call__(self, df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
        from narwhals._expression_parsing import parse_into_expr
        from narwhals._pandas_like.namespace import PandasLikeNamespace

        plx = PandasLikeNamespace(
            implementation=self._implementation, backend_version=self._backend_version
        )

        condition = parse_into_expr(self._condition, namespace=plx)._call(df)[0]  # type: ignore[arg-type]
        try:
            value_series = parse_into_expr(self._then_value, namespace=plx)._call(df)[0]  # type: ignore[arg-type]
        except TypeError:
            # `self._then_value` is a scalar and can't be converted to an expression
            value_series = condition.__class__._from_iterable(  # type: ignore[call-arg]
                [self._then_value] * len(condition),
                name="literal",
                index=condition._native_series.index,
                implementation=self._implementation,
                backend_version=self._backend_version,
            )
        value_series = cast(PandasLikeSeries, value_series)

        set_then = condition
        set_then_native = set_then._native_series
        above_already_set = parse_into_expr(self._above_already_set, namespace=plx)._call(
            df  # type: ignore[arg-type]
        )[0]

        value_series_native = value_series._native_series

        above_result = self._above_then._call(df)[0]
        above_result_native = above_result._native_series
        set_then_native = set_then._native_series
        above_already_set_native = above_already_set._native_series
        if self._otherwise_value is None:
            return [
                above_result._from_native_series(
                    value_series_native.where(
                        ~above_already_set_native & set_then_native, above_result_native
                    )
                )
            ]

        try:
            otherwise_series = parse_into_expr(
                self._otherwise_value, namespace=plx
            )._call(df)[0]  # type: ignore[arg-type]
        except TypeError:
            # `self._otherwise_value` is a scalar and can't be converted to an expression
            otherwise_series = condition.__class__._from_iterable(  # type: ignore[call-arg]
                [self._otherwise_value] * len(condition),
                name="literal",
                index=condition._native_series.index,
                implementation=self._implementation,
                backend_version=self._backend_version,
            )
        otherwise_series = cast(PandasLikeSeries, otherwise_series)
        return [
            above_result.zip_with(
                above_already_set, value_series.zip_with(set_then, otherwise_series)
            )
        ]

    def then(self, value: Any) -> PandasChainedThen:
        self._then_value = value
        return PandasChainedThen(
            self,
            depth=self._depth,
            implementation=self._implementation,
            function_name="chainedwhen",
            root_names=None,
            output_names=None,
            backend_version=self._backend_version,
        )


class PandasChainedThen(PandasLikeExpr):
    def __init__(
        self,
        call: PandasChainedWhen,
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        implementation: Implementation,
        backend_version: tuple[int, ...],
    ) -> None:
        self._implementation = implementation
        self._backend_version = backend_version

        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names

    def when(
        self,
        *predicates: IntoPandasLikeExpr,
    ) -> PandasChainedWhen:
        plx = PandasLikeNamespace(self._implementation, self._backend_version)
        if predicates:
            condition = plx.all_horizontal(*predicates)
        else:
            msg = "at least one predicate needs to be provided"
            raise TypeError(msg)
        return PandasChainedWhen(
            self,
            condition,
            depth=self._depth + 1,
            implementation=self._implementation,
            backend_version=self._backend_version,
        )

    def otherwise(self, value: Any) -> PandasChainedThen:
        self._call._otherwise_value = value  # type: ignore[attr-defined]
        self._function_name = "chainedwhenotherwise"
        return self
