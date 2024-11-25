from __future__ import annotations

from typing import Callable


class FormattedKeyError(KeyError):
    """KeyError with formatted error message.

    Python's `KeyError` has special casing around formatting
    (see https://bugs.python.org/issue2651). Use this class when the error
    message has newlines and other special format characters.
    Needed by https://github.com/tensorflow/tensorflow/issues/36857.
    """

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return self.message


class ColumnNotFoundError(FormattedKeyError):
    """Exception raised when column name isn't present."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

    @classmethod
    def from_missing_and_available_column_names(
        cls, missing_columns: list[str], available_columns: list[str]
    ) -> ColumnNotFoundError:
        message = (
            f"The following columns were not found: {missing_columns}"
            f"\n\nHint: Did you mean one of these columns: {available_columns}?"
        )
        return ColumnNotFoundError(message)


class InvalidOperationError(Exception):
    """Exception raised during invalid operations."""


class InvalidIntoExprError(TypeError):
    """Exception raised when object can't be converted to expression."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

    @classmethod
    def from_invalid_type(cls, invalid_type: type) -> InvalidIntoExprError:
        message = (
            f"Expected an object which can be converted into an expression, got {invalid_type}\n\n"
            "Hint:\n"
            "- if you were trying to select a column which does not have a string\n"
            "  column name, then you should explicitly use `nw.col`.\n"
            "  For example, `df.select(nw.col(0))` if you have a column named `0`.\n"
            "- if you were trying to create a new literal column, then you \n"
            "  should explicitly use `nw.lit`.\n"
            "  For example, `df.select(nw.lit(0))` if you want to create a new\n"
            "  column with literal value `0`."
        )
        return InvalidIntoExprError(message)


class NarwhalsUnstableWarning(UserWarning):
    """Warning issued when a method or function is considered unstable in the stable api."""


class InvalidIsDataFrameSeriesError(TypeError):
    """Exception for `is_*_dataframe` or `is_*_series` methods.

    Raised when `nw.DataFrame` or `nw.Series` is passed.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

    @classmethod
    def from_input_func_name_type(
        cls,
        input_type: type,
        function: Callable,  # type: ignore[type-arg]
    ) -> InvalidIsDataFrameSeriesError:
        from narwhals.series import Series

        obj_name = "ser" if input_type == Series else "df"
        function_name = function.__name__
        message = (
            f"You passed a `{input_type}` to `{function_name}`.\n\n"
            f"Hint: Instead of e.g. `{function_name}({obj_name})`, "
            f"did you mean `{function_name}({obj_name}.to_native())`?"
        )
        return InvalidIsDataFrameSeriesError(message)
