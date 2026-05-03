# ruff: noqa: PYI021
"""Super secret stub for the hairy bits."""

import functools
from collections.abc import Callable
from typing import Any, TypeVar, type_check_only

from narwhals._utils import Version

_T = TypeVar("_T")

# https://github.com/microsoft/pyright/issues/10956
@type_check_only
class FromNativeDispatch(functools._SingleDispatchCallable[_T]):
    """Enforces `native` and `version` as required.

    `CompliantSeries.from_native` accepts a positional `name`, so this
    signature can describe that *and* `*DataFrame`, `*LazyFrame`.
    """
    def __call__(
        self, /, native: Any, *args: Any, version: Version, **kwds: Any
    ) -> _T: ...

def from_native_dispatch(func: Callable[..., _T]) -> FromNativeDispatch[_T]:
    """This is `@functools.singledispatch`, with a narrower signature."""
