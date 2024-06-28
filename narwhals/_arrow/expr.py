from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from narwhals._arrow.series import ArrowSeries
from narwhals._pandas_like.expr import PandasExpr

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.dataframe import PandasDataFrame


class ArrowExpr(PandasExpr):
    def __init__(  # noqa: PLR0913
        self,
        call: Callable[[PandasDataFrame], list[PandasSeries]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        implementation: str,
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._depth = depth
        self._output_names = output_names
        self._implementation = implementation

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ArrowExpr("
            f"depth={self._depth}, "
            f"function_name={self._function_name}, "
            f"root_names={self._root_names}, "
            f"output_names={self._output_names}"
        )

    @classmethod
    def from_column_names(
        cls: type[Self], *column_names: str, implementation: str
    ) -> Self:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            return [
                ArrowSeries(
                    df._dataframe[column_name],
                    implementation=df._implementation,
                    name=column_name,
                )
                for column_name in column_names
            ]

        return cls(
            func,
            depth=0,
            function_name="col",
            root_names=list(column_names),
            output_names=list(column_names),
            implementation=implementation,
        )
