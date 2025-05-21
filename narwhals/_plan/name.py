from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import Function
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from narwhals._compliant.typing import AliasName


class NameFunction(Function):
    """`polars` version [doesn't represent as `FunctionExpr`].

    Also [doesn't support serialization].

    [doesn't represent as `FunctionExpr`]: https://github.com/pola-rs/polars/blob/6df23a09a81c640c21788607611e09d9f43b1abc/crates/polars-plan/src/dsl/name.rs
    [doesn't support serialization]: https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr_dyn_fn.rs#L145-L151
    """

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        tp = type(self)
        if tp is NameFunction:
            return tp.__name__
        m: dict[type[NameFunction], str] = {
            Keep: "keep",
            Map: "map",
            Suffix: "suffix",
            Prefix: "prefix",
            ToLowercase: "to_lowercase",
            ToUppercase: "to_uppercase",
        }
        return f"name.{m[tp]}"


class Keep(NameFunction):
    """Returns ``Expr::KeepName``."""


class Map(NameFunction):
    """Returns ``Expr::RenameAlias``.

    https://github.com/pola-rs/polars/blob/6df23a09a81c640c21788607611e09d9f43b1abc/crates/polars-plan/src/dsl/name.rs#L28-L38
    """

    __slots__ = ("function",)

    function: AliasName


class Prefix(NameFunction):
    """Each of these depend on `Map`."""

    __slots__ = ("prefix",)

    prefix: str


class Suffix(NameFunction):
    __slots__ = ("suffix",)

    suffix: str


class ToLowercase(NameFunction): ...


class ToUppercase(NameFunction): ...
