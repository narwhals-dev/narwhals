from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant.typing import CompliantLazyFrameT, NativeExprT
from narwhals._typing_compat import Protocol38

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._compliant.window import WindowInputs


from narwhals._compliant.when_then import LazyWhen
from narwhals._sql.expr import SQLExprT


class SQLWhen(
    LazyWhen[CompliantLazyFrameT, NativeExprT, SQLExprT],
    Protocol38[CompliantLazyFrameT, NativeExprT, SQLExprT],
):
    def _window_function(
        self, df: CompliantLazyFrameT, window_inputs: WindowInputs[NativeExprT]
    ) -> Sequence[NativeExprT]:
        is_expr = self._condition._is_expr
        condition = self._condition.window_function(df, window_inputs)[0]
        then_ = self._then_value
        then = (
            then_.window_function(df, window_inputs)[0]
            if is_expr(then_)
            else self.lit(then_)
        )

        other_ = self._otherwise_value
        if other_ is None:
            result = self.when(condition, then)
        else:
            other = (
                other_.window_function(df, window_inputs)[0]
                if is_expr(other_)
                else self.lit(other_)
            )
            result = self.when(condition, then).otherwise(other)  # type: ignore  # noqa: PGH003
        return [result]
