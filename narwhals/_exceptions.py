from __future__ import annotations


class ColumnNotFoundError(Exception): ...


class InvalidOperationError(Exception): ...


class InvalidIntoExprError(TypeError):
    def __init__(self, invalid_type: type) -> None:
        self.message = (
            f"Expected an object which can be converted into an expression, got {invalid_type}\n\n"
            "Hint: if you were trying to select a column which does not have a string column name, then "
            "you should explicitly use `nw.col`.\nFor example, `df.select(nw.col(0))` if you have a column "
            "named `0`."
        )
        super().__init__(self.message)
