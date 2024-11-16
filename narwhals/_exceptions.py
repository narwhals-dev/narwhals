from __future__ import annotations


class FormattedKeyError(KeyError):
  """KeyError with formatted error message.
  Python's `KeyError` has special casing around formatting
  (see https://bugs.python.org/issue2651). Use this class when the error
  message has newlines and other special format characters.
  Needed by https://github.com/tensorflow/tensorflow/issues/36857.
  """
  def __init__(self, message):
    self.message = message

  def __str__(self):
    return self.message
  

class ColumnNotFoundError(FormattedKeyError): ...


class InvalidOperationError(Exception): ...
