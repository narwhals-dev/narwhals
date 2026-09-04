from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from narwhals._constants import GET_CATEGORIES_DEPRECATION_TEMPLATE
from narwhals._exceptions import issue_deprecation_warning
from narwhals._expression_parsing import ExprKind, ExprNode
from narwhals._utils import Version
from narwhals.dtypes import String

if TYPE_CHECKING:
    from narwhals.expr import Expr

ExprT = TypeVar("ExprT", bound="Expr")


class ExprCatNamespace(Generic[ExprT]):
    def __init__(self, expr: ExprT) -> None:
        self._expr = expr

    def get_categories(self) -> ExprT:
        """Get unique categories from column.

        Warning:
            This is deprecated and it will be removed in a future version, as Polars
            removed its own `cat.get_categories`.
            To get the distinct values present in a Categorical column, use
            [`Expr.unique`][narwhals.Expr.unique].
            For the fixed category list of an Enum, use its `dtype.categories`.

            Until it is removed, it is implemented as
            `unique().drop_nulls().cast(String)`, so it returns only the values which
            are actually present, in no guaranteed order.

        Note:
            In `narwhals.stable.v1` and `narwhals.stable.v2` this stays available, and
            keeps dispatching to each backend's native implementation.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame(
            ...     {"fruits": ["apple", "mango", "mango"]}, dtype="category"
            ... )
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("fruits").cat.get_categories()).to_native()
              fruits
            0  apple
            1  mango
        """
        if self._expr._version in {Version.V1, Version.V2}:
            return self._expr._append_node(
                ExprNode(ExprKind.FILTRATION, "cat.get_categories")
            )

        issue_deprecation_warning(
            GET_CATEGORIES_DEPRECATION_TEMPLATE.format(cls="Expr"), _version="2.26.0"
        )
        return self._expr.unique().drop_nulls().cast(String)
