from __future__ import annotations

from narwhals._plan.common import Function
from narwhals._plan.options import FunctionFlags
from narwhals._plan.options import FunctionOptions


class StringFunction(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


class ConcatHorizontal(StringFunction):
    """`nw.functions.concat_str`."""

    __slots__ = ("ignore_nulls", "separator")

    separator: str
    ignore_nulls: bool

    @property
    def function_options(self) -> FunctionOptions:
        return super().function_options.with_flags(FunctionFlags.INPUT_WILDCARD_EXPANSION)


class Contains(StringFunction):
    __slots__ = ("literal",)

    literal: bool


class EndsWith(StringFunction): ...


class LenChars(StringFunction): ...


class Replace(StringFunction):
    __slots__ = ("literal",)

    literal: bool


class ReplaceAll(StringFunction):
    """`polars` uses a single node for this and `Replace`.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L65-L70
    """

    __slots__ = ("literal",)

    literal: bool


class Slice(StringFunction):
    """We're using for `Head`, `Tail` as well.

    https://github.com/dangotbanned/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L87-L89

    I don't think it's likely we'll support `Expr` as inputs for this any time soon.
    """

    __slots__ = ("length", "offset")

    offset: int
    length: int | None


class Head(StringFunction):
    __slots__ = ("n",)

    n: int


class Tail(StringFunction):
    __slots__ = ("n",)

    n: int


class Split(StringFunction): ...


class StartsWith(StringFunction): ...


class StripChars(StringFunction): ...


class ToDatetime(StringFunction):
    """`polars` uses `Strptime`.

    We've got a fairly different representation.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L112
    """

    __slots__ = ("format",)

    format: str | None


class ToLowercase(StringFunction): ...


class ToUppercase(StringFunction): ...
