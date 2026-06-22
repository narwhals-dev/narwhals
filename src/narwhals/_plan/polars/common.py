from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import narwhals.exceptions
from narwhals.exceptions import NarwhalsError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ("remap_exceptions",)


class remap_exceptions:  # noqa: N801
    """Fancy version of `catch_polars_exception`.

    Just write *potentially-raising* code `with`-in the context manager:

        with remap_exceptions():
            risky_business()  # Any native exceptions will be re-raised
                              # as their narwhals-equivalent
        business_as_usual()

    Works in a similar way to the implementation of [`suppress.__exit__`].

    See Also:
        [The `with` statement]

    [`suppress.__exit__`]: https://github.com/python/cpython/blob/fa7212b0af1c3d4e0cf8ac2ead35df3541436fb4/Lib/contextlib.py#L450-L469
    [The `with` statement]: https://docs.python.org/3/reference/compound_stmts.html#the-with-statement
    """

    __slots__ = ()

    _REMAP: Mapping[type[BaseException], type[NarwhalsError]] = {
        tp: getattr(narwhals.exceptions, tp.__name__, NarwhalsError)
        for tp in pl.exceptions.PolarsError.__subclasses__()
    }

    def __enter__(self) -> None:
        return

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _: object,
        /,
    ) -> bool | None:
        if exc_type is None or exc_value is None:
            return None
        if to_exc := remap_exceptions._REMAP.get(exc_type):
            raise to_exc(str(exc_value)) from None
        return False
