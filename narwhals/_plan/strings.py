from __future__ import annotations

from narwhals._plan.common import Function
from narwhals._plan.options import FunctionFlags
from narwhals._plan.options import FunctionOptions


# TODO @dangotbanned: repr
class StringFunction(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return "StringFunction"


class ConcatHorizontal(StringFunction):
    """`nw.functions.concat_str`."""

    __slots__ = ("ignore_nulls", "separator")

    separator: str
    ignore_nulls: bool

    @property
    def function_options(self) -> FunctionOptions:
        return super().function_options.with_flags(FunctionFlags.INPUT_WILDCARD_EXPANSION)

    def __repr__(self) -> str:
        return "str.concat_horizontal"


class Contains(StringFunction):
    __slots__ = ("literal",)

    literal: bool

    def __repr__(self) -> str:
        return "str.contains"


class EndsWith(StringFunction):
    def __repr__(self) -> str:
        return "str.ends_with"


class LenChars(StringFunction):
    def __repr__(self) -> str:
        return "str.len_chars"


class Replace(StringFunction):
    __slots__ = ("literal",)

    literal: bool

    def __repr__(self) -> str:
        return "str.replace"


class ReplaceAll(StringFunction):
    """`polars` uses a single node for this and `Replace`.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L65-L70
    """

    __slots__ = ("literal",)

    literal: bool

    def __repr__(self) -> str:
        return "str.replace_all"


class Slice(StringFunction):
    """We're using for `Head`, `Tail` as well.

    https://github.com/dangotbanned/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L87-L89

    I don't think it's likely we'll support `Expr` as inputs for this any time soon.
    """

    __slots__ = ("length", "offset")

    offset: int
    length: int | None

    def __repr__(self) -> str:
        return "str.slice"


class Head(StringFunction):
    __slots__ = ("n",)

    n: int

    def __repr__(self) -> str:
        return "str.head"


class Tail(StringFunction):
    __slots__ = ("n",)

    n: int

    def __repr__(self) -> str:
        return "str.tail"


class Split(StringFunction):
    def __repr__(self) -> str:
        return "str.split"


class StartsWith(StringFunction):
    def __repr__(self) -> str:
        return "str.startswith"


class StripChars(StringFunction):
    def __repr__(self) -> str:
        return "str.strip_chars"


class ToDatetime(StringFunction):
    """`polars` uses `Strptime`.

    We've got a fairly different representation.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L112
    """

    __slots__ = ("format",)

    format: str | None


class ToLowercase(StringFunction):
    def __repr__(self) -> str:
        return "str.to_lowercase"


class ToUppercase(StringFunction):
    def __repr__(self) -> str:
        return "str.to_uppercase"
