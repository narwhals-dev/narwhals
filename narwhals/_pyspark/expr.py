from __future__ import annotations

import operator
from copy import copy
from typing import TYPE_CHECKING
from typing import Callable

from narwhals._pyspark.utils import get_column_name
from narwhals._pyspark.utils import maybe_evaluate

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
        dtypes: DTypes,
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names
        self._dtypes = dtypes

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> PySparkNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._pyspark.namespace import PySparkNamespace

        return PySparkNamespace(dtypes=self._dtypes)

    @classmethod
    def from_column_names(cls: type[Self], *column_names: str, dtypes: DTypes) -> Self:
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
            dtypes=dtypes,
        )

    def _from_call(
        self,
        call: Callable[..., Column],
        expr_name: str,
        *args: PySparkExpr,
        **kwargs: PySparkExpr,
    ) -> Self:
        def func(df: PySparkLazyFrame) -> list[Column]:
            results = []
            inputs = self._call(df)
            _args = [maybe_evaluate(df, arg) for arg in args]
            _kwargs = {key: maybe_evaluate(df, value) for key, value in kwargs.items()}
            for _input in inputs:
                # For safety, _from_call should not change the name of the column
                input_col_name = get_column_name(df, _input)
                column_result = call(_input, *_args, **_kwargs).alias(input_col_name)
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
                    # TODO(unassigned): increase coverage
                    root_names = None
                    output_names = None
                    break
            elif root_names is None:  # pragma: no cover
                # TODO(unassigned): increase coverage
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
            dtypes=self._dtypes,
        )

    def __and__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.and_, "__and__", other)

    def __add__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.add, "__add__", other)

    def __radd__(self, other: PySparkExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__radd__(other), "__radd__", other
        )

    def __sub__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.sub, "__sub__", other)

    def __rsub__(self, other: PySparkExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__rsub__(other), "__rsub__", other
        )

    def __mul__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.mul, "__mul__", other)

    def __rmul__(self, other: PySparkExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__rmul__(other), "__rmul__", other
        )

    def __truediv__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.truediv, "__truediv__", other)

    def __rtruediv__(self, other: PySparkExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__rtruediv__(other), "__rtruediv__", other
        )

    def __floordiv__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.floordiv, "__floordiv__", other)

    def __rfloordiv__(self, other: PySparkExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__rfloordiv__(other), "__rfloordiv__", other
        )

    def __mod__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.mod, "__mod__", other)

    def __rmod__(self, other: PySparkExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__rmod__(other), "__rmod__", other
        )

    def __pow__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.pow, "__pow__", other)

    def __rpow__(self, other: PySparkExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__rpow__(other), "__rpow__", other
        )

    def __lt__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.lt, "__lt__", other)

    def __gt__(self, other: PySparkExpr) -> Self:
        return self._from_call(operator.gt, "__gt__", other)

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
            dtypes=self._dtypes,
        )

    def count(self) -> Self:
        def _count(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812

            return F.count(_input)

        return self._from_call(_count, "count")

    def len(self) -> Self:
        def _len(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812
            from pyspark.sql.window import Window

            return F.size(_input).over(Window.partitionBy())

        return self._from_call(_len, "len")

    def max(self) -> Self:
        def _max(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812
            from pyspark.sql.window import Window

            return F.max(_input).over(Window.partitionBy())

        return self._from_call(_max, "max")

    def mean(self) -> Self:
        def _mean(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812
            from pyspark.sql.window import Window

            return F.mean(_input).over(Window.partitionBy())

        return self._from_call(_mean, "mean")

    def min(self) -> Self:
        def _min(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812
            from pyspark.sql.window import Window

            return F.min(_input).over(Window.partitionBy())

        return self._from_call(_min, "min")

    def std(self, ddof: int = 1) -> Self:
        def std(_input: Column) -> Column:
            from pyspark.sql import functions as F  # noqa: N812
            from pyspark.sql.window import Window

            return F.stddev(_input).over(Window.partitionBy())

        _ = ddof
        return self._from_call(std, "std")