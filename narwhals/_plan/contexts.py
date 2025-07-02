from __future__ import annotations

import enum

__all__ = ["ExprContext"]


class ExprContext(enum.Enum):
    """A [context] to evaluate expressions in.

    [context]: https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/#contexts
    """

    SELECT = "select"
    """The output schema has the same order and length as the (expanded) input expressions.

    That order is determined during expansion of selectors in an earlier step.
    """

    WITH_COLUMNS = "with_columns"
    """The output schema *derives from* the input schema, but *may* produce a different shape.

    - Expressions producing **new names** are appended to the end of the schema
    - Expressions producing **existing names** will replace the existing column positionally
    """

    def is_select(self) -> bool:
        return self is ExprContext.SELECT

    def is_with_columns(self) -> bool:
        return self is ExprContext.WITH_COLUMNS
