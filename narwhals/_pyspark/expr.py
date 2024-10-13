from __future__ import annotations

import operator
from copy import copy
from typing import TYPE_CHECKING
from typing import Callable

from narwhals._pyspark.utils import get_column_name
from narwhals._pyspark.utils import maybe_evaluate
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from pyspark.sql import Column
    from typing_extensions import Self

    from narwhals._pyspark.dataframe import PySparkLazyFrame
    from narwhals._pyspark.namespace import PySparkNamespace
    from narwhals.typing import DTypes


class PySparkExpr:
    def __init__(
        self,
        call: Callable[[PySparkLazyFrame], list[Column]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        # Whether the expression is a length-1 Column resulting from
        # a reduction, such as `nw.col('a').sum()`
        returns_scalar: bool,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names
        self._returns_scalar = returns_scalar
        self._backend_version = backend_version
        self._dtypes = dtypes

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> PySparkNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._pyspark.namespace import PySparkNamespace

        return PySparkNamespace(
            backend_version=self._backend_version, dtypes=self._dtypes
        )

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> Self:
        def func(df: PySparkLazyFrame) -> list[Column]:
            from pyspark.sql import functions as F  # noqa: N812

            _ = df
            return [F.col(col_name) for col_name in column_names]

        return cls(
            func,
            depth=0,
            function_name="col",
            root_names=list(column_names),
            output_names=list(column_names),
            returns_scalar=False,
            backend_version=backend_version,
            dtypes=dtypes,
        )

    def _from_call(
        self,
        call: Callable[..., Column],
        expr_name: str,
        *args: PySparkExpr,
        returns_scalar: bool,
        **kwargs: PySparkExpr,
    ) -> Self:
        def func(df: PySparkLazyFrame) -> list[Column]:
            results = []
            inputs = self._call(df)
            _args = [maybe_evaluate(df, arg) for arg in args]
            _kwargs = {key: maybe_evaluate(df, value) for key, value in kwargs.items()}
            for _input in inputs:
                input_col_name = get_column_name(df, _input)
                column_result = call(_input, *_args, **_kwargs)
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
        for arg in list(args) + list(kwargs.values()):
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
            dtypes=self._dtypes,
        )

    def __add__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.add, "__add__", other, returns_scalar=False)

    def __sub__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.sub, "__sub__", other, returns_scalar=False)

    def __mul__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.mul, "__mul__", other, returns_scalar=False)

    def __lt__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.lt, "__lt__", other, returns_scalar=False)

    def __gt__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.gt, "__gt__", other, returns_scalar=False)

    def alias(self, name: str) -> Self:
        def _alias(df: PySparkLazyFrame) -> list[Column]:
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
            dtypes=self._dtypes,
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

    def std(self, ddof: int = 1) -> Self:
        import numpy as np  # ignore-banned-import

        def _std(_input: Column) -> Column:  # pragma: no cover
            if self._backend_version < (3, 4) or parse_version(np.__version__) > (2, 0):
                from pyspark.sql.functions import stddev

                _ = ddof
                return stddev(_input)
            from pyspark.pandas.spark.functions import stddev

            return stddev(_input, ddof=ddof)

        return self._from_call(_std, "std", returns_scalar=True)
