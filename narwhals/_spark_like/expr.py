from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence

from narwhals._spark_like.utils import get_column_name
from narwhals._spark_like.utils import maybe_evaluate
from narwhals.typing import CompliantExpr
from narwhals.utils import Implementation
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from pyspark.sql import Column
    from typing_extensions import Self

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.namespace import SparkLikeNamespace
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class SparkLikeExpr(CompliantExpr["Column"]):
    _implementation = Implementation.PYSPARK

    def __init__(
        self,
        call: Callable[[SparkLikeLazyFrame], list[Column]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        # Whether the expression is a length-1 Column resulting from
        # a reduction, such as `nw.col('a').sum()`
        returns_scalar: bool,
        backend_version: tuple[int, ...],
        version: Version,
        kwargs: dict[str, Any],
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names
        self._returns_scalar = returns_scalar
        self._backend_version = backend_version
        self._version = version
        self._kwargs = kwargs

    def __call__(self, df: SparkLikeLazyFrame) -> Sequence[Column]:
        return self._call(df)

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> SparkLikeNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._spark_like.namespace import SparkLikeNamespace

        return SparkLikeNamespace(  # type: ignore[abstract]
            backend_version=self._backend_version, version=self._version
        )

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(_: SparkLikeLazyFrame) -> list[Column]:
            from pyspark.sql import functions as F  # noqa: N812

            return [F.col(col_name) for col_name in column_names]

        return cls(
            func,
            depth=0,
            function_name="col",
            root_names=list(column_names),
            output_names=list(column_names),
            returns_scalar=False,
            backend_version=backend_version,
            version=version,
            kwargs={},
        )

    def _from_call(
        self,
        call: Callable[..., Column],
        expr_name: str,
        *,
        returns_scalar: bool,
        **kwargs: Any,
    ) -> Self:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            results = []
            inputs = self._call(df)
            _kwargs = {key: maybe_evaluate(df, value) for key, value in kwargs.items()}
            for _input in inputs:
                input_col_name = get_column_name(df, _input)
                column_result = call(_input, **_kwargs)
                if not returns_scalar:
                    column_result = column_result.alias(input_col_name)
                results.append(column_result)
            return results

        # Try tracking root and output names by combining them from all
        # expressions appearing in args and kwargs. If any anonymous
        # expression appears (e.g. nw.all()), then give up on tracking root names
        # and just set it to None.
        root_names = copy(self._root_names)
        output_names = self._output_names
        for arg in list(kwargs.values()):
            if root_names is not None and isinstance(arg, self.__class__):
                if arg._root_names is not None:
                    root_names.extend(arg._root_names)
                else:  # pragma: no cover
                    root_names = None
                    output_names = None
                    break
            elif root_names is None:
                output_names = None
                break

        if not (
            (output_names is None and root_names is None)
            or (output_names is not None and root_names is not None)
        ):  # pragma: no cover
            msg = "Safety assertion failed, please report a bug to https://github.com/narwhals-dev/narwhals/issues"
            raise AssertionError(msg)

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{expr_name}",
            root_names=root_names,
            output_names=output_names,
            returns_scalar=self._returns_scalar or returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
            kwargs=kwargs,
        )

    def __add__(self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input + other,
            "__add__",
            other=other,
            returns_scalar=False,
        )

    def __sub__(self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input - other,
            "__sub__",
            other=other,
            returns_scalar=False,
        )

    def __mul__(self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input * other,
            "__mul__",
            other=other,
            returns_scalar=False,
        )

    def __lt__(self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input < other,
            "__lt__",
            other=other,
            returns_scalar=False,
        )

    def __gt__(self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input > other,
            "__gt__",
            other=other,
            returns_scalar=False,
        )

    def alias(self, name: str) -> Self:
        def _alias(df: SparkLikeLazyFrame) -> list[Column]:
            return [col.alias(name) for col in self._call(df)]

        # Define this one manually, so that we can
        # override `output_names` and not increase depth
        return self.__class__(
            _alias,
            depth=self._depth,
            function_name=self._function_name,
            root_names=self._root_names,
            output_names=[name],
            returns_scalar=self._returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={**self._kwargs, "name": name},
        )

    def count(self) -> Self:
        def _count(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.count(_input)

        return self._from_call(_count, "count", returns_scalar=True)

    def max(self) -> Self:
        def _max(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.max(_input)

        return self._from_call(_max, "max", returns_scalar=True)

    def mean(self) -> Self:
        def _mean(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.mean(_input)

        return self._from_call(_mean, "mean", returns_scalar=True)

    def min(self) -> Self:
        def _min(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.min(_input)

        return self._from_call(_min, "min", returns_scalar=True)

    def sum(self) -> Self:
        def _sum(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.sum(_input)

        return self._from_call(_sum, "sum", returns_scalar=True)

    def std(self: Self, ddof: int) -> Self:
        from functools import partial

        import numpy as np  # ignore-banned-import

        from narwhals._spark_like.utils import _std

        func = partial(
            _std,
            ddof=ddof,
            backend_version=self._backend_version,
            np_version=parse_version(np.__version__),
        )

        return self._from_call(func, "std", returns_scalar=True, ddof=ddof)

    def var(self: Self, ddof: int) -> Self:
        from functools import partial

        import numpy as np  # ignore-banned-import

        from narwhals._spark_like.utils import _var

        func = partial(
            _var,
            ddof=ddof,
            backend_version=self._backend_version,
            np_version=parse_version(np.__version__),
        )

        return self._from_call(func, "var", returns_scalar=True, ddof=ddof)

    def abs(self) -> Self:
        def _abs(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.abs(_input)

        return self._from_call(_abs, "abs", returns_scalar=self._returns_scalar)

    def all(self) -> Self:
        def _all(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.min(
                F.when(_input.isNull() | ~_input, value=False).otherwise(value=True)
            )

        return self._from_call(_all, "all", returns_scalar=True)

    def any(self) -> Self:
        def _any(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.max(
                F.when(_input.isNull() | ~_input, value=False).otherwise(value=True)
            )

        return self._from_call(_any, "any", returns_scalar=True)

    def arg_true(self) -> Self:
        def _arg_true(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.array_position(F.collect_list(_input), value=True)

        return self._from_call(_arg_true, "arg_true", returns_scalar=True)

    def clip(
        self,
        lower_bound: Any | None = None,
        upper_bound: Any | None = None,
    ) -> Self:
        def _clip(_input: Column, lower_bound: Any, upper_bound: Any) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            result = _input
            if lower_bound is not None:
                result = F.greatest(result, lower_bound)
            if upper_bound is not None:
                result = F.least(result, upper_bound)
            return result

        return self._from_call(
            _clip,
            "clip",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            returns_scalar=self._returns_scalar,
        )

    def drop_nulls(self) -> Self:
        def _drop_nulls(_input: Column) -> Column:
            return _input.dropna()

        return self._from_call(
            _drop_nulls, "drop_nulls", returns_scalar=self._returns_scalar
        )

    def filter(self, predicate: Any) -> Self:
        def _filter(_input: Column, predicate: Any) -> Column:
            return _input.filter(predicate)

        return self._from_call(
            _filter,
            "filter",
            predicate=predicate,
            returns_scalar=self._returns_scalar,
        )

    def is_between(
        self,
        lower_bound: Any,
        upper_bound: Any,
        closed: str = "both",
    ) -> Self:
        def _is_between(_input: Column, lower_bound: Any, upper_bound: Any) -> Column:
            if closed == "both":
                return (_input >= lower_bound) & (_input <= upper_bound)
            if closed == "neither":
                return (_input > lower_bound) & (_input < upper_bound)
            if closed == "left":
                return (_input >= lower_bound) & (_input < upper_bound)
            return (_input > lower_bound) & (_input <= upper_bound)

        return self._from_call(
            _is_between,
            "is_between",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            returns_scalar=self._returns_scalar,
        )

    def is_duplicated(self) -> Self:
        def _is_duplicated(_input: Column) -> Column:
            from pyspark.sql import Window
            from pyspark.sql import functions as F  # noqa: N812

            return F.count(_input).over(Window.partitionBy(_input)) > 1

        return self._from_call(
            _is_duplicated, "is_duplicated", returns_scalar=self._returns_scalar
        )

    def is_finite(self) -> Self:
        def _is_finite(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return ~F.isnan(_input) & ~F.isnull(_input)

        return self._from_call(
            _is_finite, "is_finite", returns_scalar=self._returns_scalar
        )

    def is_in(self, values: Sequence[Any]) -> Self:
        def _is_in(_input: Column, values: Sequence[Any]) -> Column:
            return _input.isin(values)

        return self._from_call(
            _is_in,
            "is_in",
            values=values,
            returns_scalar=self._returns_scalar,
        )

    def is_nan(self) -> Self:
        def _is_nan(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.isnan(_input)

        return self._from_call(_is_nan, "is_nan", returns_scalar=self._returns_scalar)

    def is_unique(self) -> Self:
        def _is_unique(_input: Column) -> Column:
            from pyspark.sql import Window
            from pyspark.sql import functions as F  # noqa: N812

            return F.count(_input).over(Window.partitionBy(_input)) == 1

        return self._from_call(
            _is_unique, "is_unique", returns_scalar=self._returns_scalar
        )

    def len(self) -> Self:
        def _len(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.count(_input)

        return self._from_call(_len, "len", returns_scalar=True)

    def map_batches(self, func: Callable[[Any], Any]) -> Self:
        def _map_batches(_input: Column, func: Callable[[Any], Any]) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.transform(_input, func)

        return self._from_call(
            _map_batches,
            "map_batches",
            func=func,
            returns_scalar=self._returns_scalar,
        )

    def median(self) -> Self:
        def _median(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.percentile_approx(_input, 0.5)

        return self._from_call(_median, "median", returns_scalar=True)

    def mode(self) -> Self:
        def _mode(_input: Column) -> Column:
            from pyspark.sql import Window
            from pyspark.sql import functions as F  # noqa: N812

            w = Window.orderBy(F.count(_input).desc())
            return F.first(_input).over(w)

        return self._from_call(_mode, "mode", returns_scalar=True)

    def n_unique(self) -> Self:
        def _n_unique(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.countDistinct(_input)

        return self._from_call(_n_unique, "n_unique", returns_scalar=True)

    def name(self, name: str) -> Self:
        return self.alias(name)

    def null_count(self) -> Self:
        def _null_count(_input: Column) -> Column:
            return _input.isNull().cast("long").sum()

        return self._from_call(_null_count, "null_count", returns_scalar=True)

    def over(self, partition_by: str | Sequence[str]) -> Self:
        def _over(_input: Column, partition_by: str | Sequence[str]) -> Column:
            from pyspark.sql import Window

            if isinstance(partition_by, str):
                partition_by = [partition_by]
            return _input.over(Window.partitionBy(*partition_by))

        return self._from_call(
            _over,
            "over",
            partition_by=partition_by,
            returns_scalar=self._returns_scalar,
        )

    def quantile(self, q: float) -> Self:
        def _quantile(_input: Column, q: float) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.percentile_approx(_input, q)

        return self._from_call(_quantile, "quantile", q=q, returns_scalar=True)

    def replace_strict(
        self,
        old: Sequence[Any] | dict[Any, Any],
        new: Sequence[Any] | None = None,
        *,
        return_dtype: DType | None = None,
    ) -> Self:
        def _replace_strict(
            _input: Column,
            old: Sequence[Any] | dict[Any, Any],
            new: Sequence[Any] | None,
        ) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            if isinstance(old, dict):
                result = _input
                for k, v in old.items():
                    result = F.when(result == k, v).otherwise(result)
                return result

            if len(old) != len(new):  # type: ignore[arg-type]  # new may be None
                msg = "Length of replacements must match"
                raise ValueError(msg)

            result = _input
            for o, n in zip(old, new):  # type: ignore[arg-type]  # new may be None
                result = F.when(result == o, n).otherwise(result)
            return result

        return self._from_call(
            _replace_strict,
            "replace_strict",
            old=old,
            new=new,
            returns_scalar=self._returns_scalar,
        )

    def round(self, decimals: int) -> Self:
        def _round(_input: Column, decimals: int) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.round(_input, decimals)

        return self._from_call(
            _round,
            "round",
            decimals=decimals,
            returns_scalar=self._returns_scalar,
        )

    def sample(self, n: int | None = None, fraction: float | None = None) -> Self:
        def _sample(_input: Column, n: int | None, fraction: float | None) -> Column:
            if n is not None:
                return _input.sample(n=n)
            if fraction is not None:
                return _input.sample(fraction=fraction)
            msg = "Either n or fraction must be specified"
            raise ValueError(msg)

        return self._from_call(
            _sample,
            "sample",
            n=n,
            fraction=fraction,
            returns_scalar=self._returns_scalar,
        )

    def skew(self) -> Self:
        def _skew(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.skewness(_input)

        return self._from_call(_skew, "skew", returns_scalar=True)

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        def _sort(_input: Column, *, descending: bool, nulls_last: bool) -> Column:
            if descending:
                _input = _input.desc()
            return _input.nulls_last() if nulls_last else _input.nulls_first()

        return self._from_call(
            _sort,
            "sort",
            descending=descending,
            nulls_last=nulls_last,
            returns_scalar=self._returns_scalar,
        )

    def unique(self) -> Self:
        def _unique(_input: Column) -> Column:
            return _input.distinct()

        return self._from_call(_unique, "unique", returns_scalar=self._returns_scalar)
