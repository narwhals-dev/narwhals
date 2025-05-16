from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import Function
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from narwhals._compliant.typing import AliasName


class NameFunction(Function):
    """`polars` version doesn't represent in the same way here.

    https://github.com/pola-rs/polars/blob/6df23a09a81c640c21788607611e09d9f43b1abc/crates/polars-plan/src/dsl/name.rs
    """

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


class Keep(NameFunction): ...


class Map(NameFunction):
    __slots__ = ("function",)

    function: AliasName


class Prefix(NameFunction):
    __slots__ = ("prefix",)

    prefix: str


class Suffix(NameFunction):
    __slots__ = ("suffix",)

    suffix: str


class ToLowercase(NameFunction): ...


class ToUppercase(NameFunction): ...
