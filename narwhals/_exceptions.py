from __future__ import annotations


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
    def __init__(self, missing_columns: list[str], available_columns: list[str]) -> None:
        self.missing_columns = missing_columns
        self.available_columns = available_columns
        self.message = self._error_message()
        super().__init__(self.message)

    def _error_message(self) -> str:
        return (
            f"The following columns were not found: {self.missing_columns}"
            f"\n\nHint: Did you mean one of these columns: {self.available_columns}?"
        )


class InvalidOperationError(Exception): ...
