from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._compliant.dataframe import CompliantLazyFrame
from narwhals._compliant.typing import (
    CompliantExprT_contra,
    NativeExprT,
    NativeLazyFrameT,
)
from narwhals._translate import ToNarwhalsT_co
from narwhals._utils import check_columns_exist, generate_temporary_column_name
from narwhals.exceptions import InvalidOperationError, MultiOutputExpressionError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from typing_extensions import Self

    from narwhals._compliant.window import WindowInputs
    from narwhals._sql.expr import SQLExpr
    from narwhals.exceptions import ColumnNotFoundError

    Incomplete: TypeAlias = Any


class SQLLazyFrame(
    CompliantLazyFrame[CompliantExprT_contra, NativeLazyFrameT, ToNarwhalsT_co],
    Protocol[CompliantExprT_contra, NativeLazyFrameT, ToNarwhalsT_co],
):
    def _evaluate_window_expr(
        self,
        expr: SQLExpr[Self, NativeExprT],
        /,
        window_inputs: WindowInputs[NativeExprT],
    ) -> NativeExprT:
        result = expr.window_function(self, window_inputs)
        if len(result) != 1:  # pragma: no cover
            msg = "multi-output expressions not allowed in this context"
            raise MultiOutputExpressionError(msg)
        return result[0]

    def _evaluate_single_output_expr(
        self, expr: SQLExpr[Self, NativeExprT], /
    ) -> NativeExprT:
        result = expr(self)
        if len(result) != 1:  # pragma: no cover
            msg = "multi-output expressions not allowed in this context"
            raise MultiOutputExpressionError(msg)
        return result[0]

    def _check_columns_exist(self, subset: Sequence[str]) -> ColumnNotFoundError | None:
        return check_columns_exist(subset, available=self.columns)

    def _validate_explode_columns(self, columns: Sequence[str]) -> str:
        """Validate `explode` inputs and return the single column to explode.

        The exploded columns must be of `List` dtype, and lazy SQL backends only
        support exploding one at a time, since they cannot guarantee that several
        exploded columns have matching element counts.
        """
        list_dtype = self._version.dtypes.List
        schema = self.collect_schema()
        for name in columns:
            if (dtype := schema[name]) != list_dtype:
                msg = (
                    f"`explode` operation not supported for dtype `{dtype}`, "
                    "expected List type"
                )
                raise InvalidOperationError(msg)
        if len(columns) != 1:
            msg = (
                "Exploding on multiple columns is not supported for lazy backends, since "
                "we cannot guarantee that the exploded columns have matching element counts."
            )
            raise NotImplementedError(msg)
        return columns[0]

    def _filter(self, predicate: CompliantExprT_contra) -> Self: ...

    def filter(self, predicate: CompliantExprT_contra) -> Self:
        if not predicate._metadata.is_elementwise:
            # add the temporary column, filter on it, then drop it
            tmp_col = generate_temporary_column_name(8, self.columns, prefix="filter")
            ns = self.__narwhals_namespace__()
            lf_with_tmp = self.with_columns(predicate.alias(tmp_col))
            filtered = lf_with_tmp._filter(ns.col(tmp_col))
            return filtered.drop([tmp_col], strict=False)
        return self._filter(predicate)
