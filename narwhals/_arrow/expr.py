from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence

import pyarrow.compute as pc

from narwhals._arrow.series import ArrowSeries
from narwhals._compliant import EagerExpr
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals._expression_parsing import is_scalar_like
from narwhals.exceptions import ColumnNotFoundError
from narwhals.utils import Implementation
from narwhals.utils import generate_temporary_column_name
from narwhals.utils import not_implemented

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals._expression_parsing import ExprMetadata
    from narwhals.utils import Version
    from narwhals.utils import _FullContext


class ArrowExpr(EagerExpr["ArrowDataFrame", ArrowSeries]):
    _implementation: Implementation = Implementation.PYARROW

    def __init__(
        self: Self,
        call: Callable[[ArrowDataFrame], Sequence[ArrowSeries]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[ArrowDataFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        backend_version: tuple[int, ...],
        version: Version,
        call_kwargs: dict[str, Any] | None = None,
        implementation: Implementation | None = None,
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._depth = depth
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._backend_version = backend_version
        self._version = version
        self._call_kwargs = call_kwargs or {}
        self._metadata: ExprMetadata | None = None

    @classmethod
    def from_column_names(
        cls: type[Self],
        evaluate_column_names: Callable[[ArrowDataFrame], Sequence[str]],
        /,
        *,
        context: _FullContext,
        function_name: str = "",
    ) -> Self:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            try:
                return [
                    ArrowSeries(
                        df.native[column_name],
                        name=column_name,
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
                    missing_columns=missing_columns, available_columns=df.columns
                ) from e

        return cls(
            func,
            depth=0,
            function_name=function_name,
            evaluate_output_names=evaluate_column_names,
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
        )

    @classmethod
    def from_column_indices(
        cls: type[Self], *column_indices: int, context: _FullContext
    ) -> Self:
        from narwhals._arrow.series import ArrowSeries

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            return [
                ArrowSeries(
                    df.native[column_index],
                    name=df.native.column_names[column_index],
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
            backend_version=context._backend_version,
            version=context._version,
        )

    def __narwhals_namespace__(self: Self) -> ArrowNamespace:
        from narwhals._arrow.namespace import ArrowNamespace

        return ArrowNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def __narwhals_expr__(self: Self) -> None: ...

    def _reuse_series_extra_kwargs(
        self, *, returns_scalar: bool = False
    ) -> dict[str, Any]:
        return {"_return_py_scalar": False} if returns_scalar else {}

    def cum_sum(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_sum", reverse=reverse)

    def shift(self: Self, n: int) -> Self:
        return self._reuse_series("shift", n=n)

    def over(
        self: Self,
        partition_by: Sequence[str],
        order_by: Sequence[str] | None,
    ) -> Self:
        assert self._metadata is not None  # noqa: S101
        if partition_by and not is_scalar_like(self._metadata.kind):
            msg = "Only aggregation or literal operations are supported in grouped `over` context for PyArrow."
            raise NotImplementedError(msg)

        if not partition_by:
            # e.g. `nw.col('a').cum_sum().order_by(key)`
            # which we can always easily support, as it doesn't require grouping.
            assert order_by is not None  # help type checkers  # noqa: S101

            def func(df: ArrowDataFrame) -> Sequence[ArrowSeries]:
                token = generate_temporary_column_name(8, df.columns)
                df = df.with_row_index(token).sort(
                    *order_by, descending=False, nulls_last=False
                )
                result = self(df.drop([token], strict=True))
                # TODO(marco): is there a way to do this efficiently without
                # doing 2 sorts? Here we're sorting the dataframe and then
                # again calling `sort_indices`. `ArrowSeries.scatter` would also sort.
                sorting_indices = pc.sort_indices(df[token].native)  # type: ignore[call-overload]
                return [
                    ser._from_native_series(pc.take(ser.native, sorting_indices))
                    for ser in result
                ]
        else:

            def func(df: ArrowDataFrame) -> Sequence[ArrowSeries]:
                output_names, aliases = evaluate_output_names_and_aliases(self, df, [])
                if overlap := set(output_names).intersection(partition_by):
                    # E.g. `df.select(nw.all().sum().over('a'))`. This is well-defined,
                    # we just don't support it yet.
                    msg = (
                        f"Column names {overlap} appear in both expression output names and in `over` keys.\n"
                        "This is not yet supported."
                    )
                    raise NotImplementedError(msg)

                tmp = df.group_by(*partition_by, drop_null_keys=False).agg(self)
                tmp = df.simple_select(*partition_by).join(
                    tmp,
                    how="left",
                    left_on=partition_by,
                    right_on=partition_by,
                    suffix="_right",
                )
                return [tmp[alias] for alias in aliases]

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=self._function_name + "->over",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def cum_count(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_count", reverse=reverse)

    def cum_min(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_min", reverse=reverse)

    def cum_max(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_max", reverse=reverse)

    def cum_prod(self: Self, *, reverse: bool) -> Self:
        return self._reuse_series("cum_prod", reverse=reverse)

    def rank(
        self: Self,
        method: Literal["average", "min", "max", "dense", "ordinal"],
        *,
        descending: bool,
    ) -> Self:
        return self._reuse_series("rank", method=method, descending=descending)

    ewm_mean = not_implemented()
