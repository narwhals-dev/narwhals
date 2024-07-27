from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from narwhals.dependencies import get_dask_expr

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._dask.dataframe import DaskLazyFrame

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
            for _input in inputs:
                _args = [maybe_evaluate(df, x) for x in args]
                _kwargs = {
                    key: maybe_evaluate(df, value) for key, value in kwargs.items()
                }
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
            results = []
            inputs = self._call(df)
            for _input in inputs:
                result = _input.rename(name)
                results.append(result)
            return results

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

    def __mul__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mul__(other),
            "__mul__",
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
