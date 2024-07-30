from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from narwhals.dependencies import get_dask
from narwhals.dependencies import get_dask_expr

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._dask.dataframe import DaskLazyFrame
    from narwhals._dask.namespace import DaskNamespace

from narwhals._dask.utils import maybe_evaluate


class DaskExpr:
    def __init__(
        self,
        # callable from DaskLazyFrame to list of (native) Dask Series
        call: Callable[[DaskLazyFrame], Any],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        backend_version: tuple[int, ...],
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names
        self._backend_version = backend_version

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> DaskNamespace:  # pragma: no cover
        from narwhals._dask.namespace import DaskNamespace

        return DaskNamespace(backend_version=self._backend_version)

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[Any]:
            return [
                df._native_dataframe.loc[:, column_name] for column_name in column_names
            ]

        return cls(
            func,
            depth=0,
            function_name="col",
            root_names=list(column_names),
            output_names=list(column_names),
            backend_version=backend_version,
        )

    def _from_call(
        self,
        # callable from DaskLazyFrame to list of (native) Dask Series
        call: Any,
        expr_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[Any]:
            results = []
            inputs = self._call(df)
            _args = [maybe_evaluate(df, x) for x in args]
            _kwargs = {key: maybe_evaluate(df, value) for key, value in kwargs.items()}
            for _input in inputs:
                result = call(_input, *_args, **_kwargs)
                if isinstance(result, get_dask_expr()._collection.Series):
                    result = result.rename(_input.name)
                results.append(result)
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
            backend_version=self._backend_version,
        )

    def alias(self, name: str) -> Self:
        def func(df: DaskLazyFrame) -> list[Any]:
            inputs = self._call(df)
            return [_input.rename(name) for _input in inputs]

        return self.__class__(
            func,
            depth=self._depth,
            function_name=self._function_name,
            root_names=self._root_names,
            output_names=[name],
            backend_version=self._backend_version,
        )

    def __add__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__add__(other),
            "__add__",
            other,
        )

    def __sub__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__sub__(other),
            "__sub__",
            other,
        )

    def __mul__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mul__(other),
            "__mul__",
            other,
        )

    def __ge__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__ge__(other),
            "__ge__",
            other,
        )

    def __gt__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__gt__(other),
            "__gt__",
            other,
        )

    def __le__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__le__(other),
            "__le__",
            other,
        )

    def __lt__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__lt__(other),
            "__lt__",
            other,
        )

    def __and__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__and__(other),
            "__and__",
            other,
        )

    def mean(self) -> Self:
        return self._from_call(
            lambda _input: _input.mean(),
            "mean",
        )

    def shift(self, n: int) -> Self:
        return self._from_call(
            lambda _input, n: _input.shift(n),
            "shift",
            n,
        )

    def cum_sum(self) -> Self:
        return self._from_call(
            lambda _input: _input.cumsum(),
            "cum_sum",
        )

    def is_between(
        self,
        lower_bound: Any,
        upper_bound: Any,
        closed: str = "both",
    ) -> Self:
        return self._from_call(
            lambda _input, lower_bound, upper_bound, closed: _input.between(
                lower_bound,
                upper_bound,
                closed,
            ),
            "is_between",
            lower_bound,
            upper_bound,
            closed,
        )

    def sum(self) -> Self:
        return self._from_call(
            lambda _input: _input.sum(),
            "sum",
        )

    def fill_null(self, value: Any) -> DaskExpr:
        return self._from_call(lambda _input, _val: _input.fillna(_val), "fillna", value)

    @property
    def str(self: Self) -> DaskExprStringNamespace:
        return DaskExprStringNamespace(self)

    @property
    def dt(self: Self) -> DaskExprDateTimeNamespace:
        return DaskExprDateTimeNamespace(self)


class DaskExprStringNamespace:
    def __init__(self, expr: DaskExpr) -> None:
        self._expr = expr

    def starts_with(self, prefix: str) -> DaskExpr:
        return self._expr._from_call(
            lambda _input, prefix: _input.str.startswith(prefix), "starts_with", prefix
        )

    def ends_with(self, suffix: str) -> DaskExpr:
        return self._expr._from_call(
            lambda _input, suffix: _input.str.endswith(suffix), "ends_with", suffix
        )

    def contains(self, pattern: str, *, literal: bool = False) -> DaskExpr:
        return self._expr._from_call(
            lambda _input, pat, regex: _input.str.contains(pat=pat, regex=regex),
            "contains",
            pattern,
            not literal,
        )

    def slice(self, offset: int, length: int | None = None) -> DaskExpr:
        stop = offset + length if length else None
        return self._expr._from_call(
            lambda _input, start, stop: _input.str.slice(start=start, stop=stop),
            "slice",
            offset,
            stop,
        )

    def to_datetime(self, format: str | None = None) -> DaskExpr:  # noqa: A002
        return self._expr._from_call(
            lambda _input, fmt: get_dask().dataframe.to_datetime(_input, format=fmt),
            "to_datetime",
            format,
        )

    def to_uppercase(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.str.upper(),
            "to_uppercase",
        )

    def to_lowercase(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.str.lower(),
            "to_lowercase",
        )


class DaskExprDateTimeNamespace:
    def __init__(self, expr: DaskExpr) -> None:
        self._expr = expr

    def year(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.year,
            "year",
        )

    def month(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.month,
            "month",
        )

    def day(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.day,
            "day",
        )

    def hour(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.hour,
            "hour",
        )

    def minute(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.minute,
            "minute",
        )

    def second(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.second,
            "second",
        )

    def millisecond(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.microsecond // 1000,
            "millisecond",
        )

    def microsecond(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.microsecond,
            "microsecond",
        )

    def nanosecond(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.microsecond * 1000 + _input.dt.nanosecond,
            "nanosecond",
        )

    def ordinal_day(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.dayofyear,
            "ordinal_day",
        )
