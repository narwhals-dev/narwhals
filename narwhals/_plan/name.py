"""TODO: Attributes."""

from __future__ import annotations

from narwhals._plan.common import Function


class NameFunction(Function):
    """`polars` version doesn't represent in the same way here.

    https://github.com/pola-rs/polars/blob/6df23a09a81c640c21788607611e09d9f43b1abc/crates/polars-plan/src/dsl/name.rs
    """


class Keep(NameFunction): ...


class Map(NameFunction): ...


class Prefix(NameFunction): ...


class Suffix(NameFunction): ...


class ToLowercase(NameFunction): ...


class ToUppercase(NameFunction): ...
