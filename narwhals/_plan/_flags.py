from __future__ import annotations

import enum


# TODO @dangotbanned: rename a loads of other things
# - would be nice to have `FunctionExpr.flags`, `Function.flags`
class FunctionFlags(enum.Flag):
    """Properties of functions."""

    # TODO @dangotbanned: Experiment with using `0` instead of `1`
    # https://docs.python.org/3/howto/enum.html#flag
    DEFAULT = 1 << 0
    REDUCE_EXPANSION = 1 << 2
    AGGREGATION = 1 << 3
    ROW_SEPARABLE = 1 << 6
    LENGTH_PRESERVING = 1 << 7

    # NOTE: Placeholder for 1 or more supertyping flags
    # `SUPERTYPING = ...`
    # (Mirroring the `rust` version created more problems than it solved)
    # https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/plans/options.rs#L174
    # https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/plans/options.rs#L192-L195
    # https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/plans/options.rs#L263-L265
    # https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-core/src/utils/supertype.rs#L101-L138

    ELEMENTWISE = ROW_SEPARABLE | LENGTH_PRESERVING
    HORIZONTAL = REDUCE_EXPANSION | ELEMENTWISE

    def __str__(self) -> str:
        # NOTE: Appeared as `self.flags` in:
        #   `f"{type(self).__name__}(flags='{self.flags}')"``
        name = self.name or "<FUNCTION_FLAGS_UNKNOWN>"
        return name.replace("|", " | ")

    def with_udf(self, *, is_elementwise: bool, returns_scalar: bool) -> FunctionFlags:
        # NOTE: Want this to be checked on `self.__class__`, since all operators go through that
        # https://docs.python.org/3/howto/enum.html#duplicatefreeenum
        opts = self
        if is_elementwise:
            opts |= FunctionFlags.ELEMENTWISE
        if returns_scalar:
            opts |= FunctionFlags.AGGREGATION
        if _INVALID in opts:
            msg = "A function cannot both return a scalar and preserve length, they are mutually exclusive."
            raise TypeError(msg)
        return opts


_INVALID = FunctionFlags.AGGREGATION | FunctionFlags.LENGTH_PRESERVING
