from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence

from duckdb import CaseExpression
from duckdb import CoalesceOperator
from duckdb import ColumnExpression
from duckdb import ConstantExpression
from duckdb import FunctionExpression

from narwhals._duckdb.expr_dt import DuckDBExprDateTimeNamespace
from narwhals._duckdb.expr_list import DuckDBExprListNamespace
from narwhals._duckdb.expr_name import DuckDBExprNameNamespace
from narwhals._duckdb.expr_str import DuckDBExprStringNamespace
from narwhals._duckdb.utils import binary_operation_returns_scalar
from narwhals._duckdb.utils import maybe_evaluate
from narwhals._duckdb.utils import narwhals_to_native_dtype
from narwhals.typing import CompliantExpr
from narwhals.utils import Implementation

if TYPE_CHECKING:
    import duckdb
    from typing_extensions import Self

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.namespace import DuckDBNamespace
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class DuckDBExpr(CompliantExpr["duckdb.Expression"]):
    _implementation = Implementation.DUCKDB

    def __init__(
        self: Self,
        call: Callable[[DuckDBLazyFrame], Sequence[duckdb.Expression]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[DuckDBLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        # Whether the expression is a length-1 Column resulting from
        # a reduction, such as `nw.col('a').sum()`
        returns_scalar: bool,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._returns_scalar = returns_scalar
        self._backend_version = backend_version
        self._version = version

    def __call__(self: Self, df: DuckDBLazyFrame) -> Sequence[duckdb.Expression]:
        return self._call(df)

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> DuckDBNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._duckdb.namespace import DuckDBNamespace

        return DuckDBNamespace(
            backend_version=self._backend_version, version=self._version
        )

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(_: DuckDBLazyFrame) -> list[duckdb.Expression]:
            return [ColumnExpression(col_name) for col_name in column_names]

        return cls(
            func,
            depth=0,
            function_name="col",
            evaluate_output_names=lambda _df: list(column_names),
            alias_output_names=None,
            returns_scalar=False,
            backend_version=backend_version,
            version=version,
        )

    @classmethod
    def from_column_indices(
        cls: type[Self],
        *column_indices: int,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            columns = df.columns

            return [ColumnExpression(columns[i]) for i in column_indices]

        return cls(
            func,
            depth=0,
            function_name="nth",
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            returns_scalar=False,
            backend_version=backend_version,
            version=version,
        )

    def _from_call(
        self: Self,
        call: Callable[..., duckdb.Expression],
        expr_name: str,
        *,
        returns_scalar: bool,
        **expressifiable_args: Self | Any,
    ) -> Self:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            native_series_list = self._call(df)
            other_native_series = {
                key: maybe_evaluate(df, value)
                for key, value in expressifiable_args.items()
            }
            return [
                call(native_series, **other_native_series)
                for native_series in native_series_list
            ]

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{expr_name}",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            returns_scalar=returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
        )

    def __and__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input & other,
            "__and__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __or__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input | other,
            "__or__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __add__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input + other,
            "__add__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __truediv__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input / other,
            "__truediv__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __floordiv__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__floordiv__(other),
            "__floordiv__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __mod__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mod__(other),
            "__mod__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __sub__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input - other,
            "__sub__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __mul__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input * other,
            "__mul__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __pow__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input**other,
            "__pow__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __lt__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input < other,
            "__lt__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __gt__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input > other,
            "__gt__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __le__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input <= other,
            "__le__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __ge__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input >= other,
            "__ge__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __eq__(self: Self, other: DuckDBExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input == other,
            "__eq__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __ne__(self: Self, other: DuckDBExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input != other,
            "__ne__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __invert__(self: Self) -> Self:
        return self._from_call(
            lambda _input: ~_input,
            "__invert__",
            returns_scalar=self._returns_scalar,
        )

    def alias(self: Self, name: str) -> Self:
        def alias_output_names(names: Sequence[str]) -> Sequence[str]:
            if len(names) != 1:
                msg = f"Expected function with single output, found output names: {names}"
                raise ValueError(msg)
            return [name]

        # Define this one manually, so that we can
        # override `output_names` and not increase depth
        return self.__class__(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=alias_output_names,
            returns_scalar=self._returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
        )

    def abs(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("abs", _input),
            "abs",
            returns_scalar=self._returns_scalar,
        )

    def mean(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("mean", _input),
            "mean",
            returns_scalar=True,
        )

    def skew(self: Self) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            count = FunctionExpression("count", _input)
            return CaseExpression(
                condition=count == 0, value=ConstantExpression(None)
            ).otherwise(
                CaseExpression(
                    condition=count == 1, value=ConstantExpression(float("nan"))
                ).otherwise(
                    CaseExpression(
                        condition=count == 2, value=ConstantExpression(0.0)
                    ).otherwise(FunctionExpression("skewness", _input))
                )
            )

        return self._from_call(func, "skew", returns_scalar=True)

    def median(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("median", _input),
            "median",
            returns_scalar=True,
        )

    def all(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("bool_and", _input),
            "all",
            returns_scalar=True,
        )

    def any(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("bool_or", _input),
            "any",
            returns_scalar=True,
        )

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            if interpolation == "linear":
                return FunctionExpression(
                    "quantile_cont", _input, ConstantExpression(quantile)
                )
            msg = "Only linear interpolation methods are supported for DuckDB quantile."
            raise NotImplementedError(msg)

        return self._from_call(
            func,
            "quantile",
            returns_scalar=True,
        )

    def clip(self: Self, lower_bound: Any, upper_bound: Any) -> Self:
        def func(
            _input: duckdb.Expression, lower_bound: Any, upper_bound: Any
        ) -> duckdb.Expression:
            return FunctionExpression(
                "greatest", FunctionExpression("least", _input, upper_bound), lower_bound
            )

        return self._from_call(
            func,
            "clip",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            returns_scalar=self._returns_scalar,
        )

    def is_between(
        self: Self,
        lower_bound: Any,
        upper_bound: Any,
        closed: Literal["left", "right", "none", "both"],
    ) -> Self:
        def func(
            _input: duckdb.Expression, lower_bound: Any, upper_bound: Any
        ) -> duckdb.Expression:
            if closed == "left":
                return (_input >= lower_bound) & (_input < upper_bound)
            elif closed == "right":
                return (_input > lower_bound) & (_input <= upper_bound)
            elif closed == "none":
                return (_input > lower_bound) & (_input < upper_bound)
            return (_input >= lower_bound) & (_input <= upper_bound)

        return self._from_call(
            func,
            "is_between",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            returns_scalar=self._returns_scalar,
        )

    def sum(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("sum", _input), "sum", returns_scalar=True
        )

    def n_unique(self: Self) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            # https://stackoverflow.com/a/79338887/4451315
            return FunctionExpression(
                "array_unique", FunctionExpression("array_agg", _input)
            ) + FunctionExpression(
                "max",
                CaseExpression(
                    condition=_input.isnotnull(), value=ConstantExpression(0)
                ).otherwise(ConstantExpression(1)),
            )

        return self._from_call(
            func,
            "n_unique",
            returns_scalar=True,
        )

    def count(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("count", _input),
            "count",
            returns_scalar=True,
        )

    def len(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("count"), "len", returns_scalar=True
        )

    def std(self: Self, ddof: int) -> Self:
        def _std(_input: duckdb.Expression, ddof: int) -> duckdb.Expression:
            n_samples = FunctionExpression("count", _input)

            return (
                FunctionExpression("stddev_pop", _input)
                * FunctionExpression("sqrt", n_samples)
                / (FunctionExpression("sqrt", (n_samples - ddof)))
            )

        return self._from_call(
            _std,
            "std",
            ddof=ddof,
            returns_scalar=True,
        )

    def var(self: Self, ddof: int) -> Self:
        def _var(_input: duckdb.Expression, ddof: int) -> duckdb.Expression:
            n_samples = FunctionExpression("count", _input)
            return FunctionExpression("var_pop", _input) * n_samples / (n_samples - ddof)

        return self._from_call(
            _var,
            "var",
            ddof=ddof,
            returns_scalar=True,
        )

    def max(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("max", _input), "max", returns_scalar=True
        )

    def min(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("min", _input), "min", returns_scalar=True
        )

    def null_count(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("sum", _input.isnull().cast("int")),
            "null_count",
            returns_scalar=True,
        )

    def is_null(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.isnull(), "is_null", returns_scalar=self._returns_scalar
        )

    def is_nan(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("isnan", _input),
            "is_nan",
            returns_scalar=self._returns_scalar,
        )

    def is_finite(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("isfinite", _input),
            "is_finite",
            returns_scalar=self._returns_scalar,
        )

    def is_in(self: Self, other: Sequence[Any]) -> Self:
        return self._from_call(
            lambda _input: _input.isin(*[ConstantExpression(x) for x in other]),
            "is_in",
            returns_scalar=self._returns_scalar,
        )

    def round(self: Self, decimals: int) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression(
                "round", _input, ConstantExpression(decimals)
            ),
            "round",
            returns_scalar=self._returns_scalar,
        )

    def fill_null(self: Self, value: Any, strategy: Any, limit: int | None) -> Self:
        if strategy is not None:
            msg = "todo"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: CoalesceOperator(_input, ConstantExpression(value)),
            "fill_null",
            returns_scalar=self._returns_scalar,
        )

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            native_dtype = narwhals_to_native_dtype(dtype, self._version)
            return _input.cast(native_dtype)

        return self._from_call(
            func,
            "cast",
            returns_scalar=self._returns_scalar,
        )

    @property
    def str(self: Self) -> DuckDBExprStringNamespace:
        return DuckDBExprStringNamespace(self)

    @property
    def dt(self: Self) -> DuckDBExprDateTimeNamespace:
        return DuckDBExprDateTimeNamespace(self)

    @property
    def name(self: Self) -> DuckDBExprNameNamespace:
        return DuckDBExprNameNamespace(self)

    @property
    def list(self: Self) -> DuckDBExprListNamespace:
        return DuckDBExprListNamespace(self)
