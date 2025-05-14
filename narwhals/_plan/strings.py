from __future__ import annotations

from narwhals._plan.common import Function


class StringFunction(Function): ...


class Contains(StringFunction): ...


class EndsWith(StringFunction): ...


class Head(StringFunction): ...


class LenChars(StringFunction): ...


class Replace(StringFunction): ...


class ReplaceAll(StringFunction):
    """`polars` uses a single node for this and `Replace`.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L65-L70
    """


class Slice(StringFunction): ...


class Split(StringFunction): ...


class StartsWith(StringFunction): ...


class StripChars(StringFunction): ...


class Tail(StringFunction): ...


class ToDatetime(StringFunction):
    """`polars` uses `Strptime`.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/strings.rs#L112
    """


class ToLowercase(StringFunction): ...


class ToUppercase(StringFunction): ...
