"""Home to `FunctionFlags`.

## Implementation Notes
- Adapted from [`plans::options::FunctionFlags`] and folds the outer struct [`FunctionOptions`] into a single enum
- [`INPUT_WILDCARD_EXPANSION`] removal
  - Defined selector expansion behavior for horizontal functions
  - *Instead*, `HorizontalFunction` is now used for that
- There will be another concept to integrate (dependent on [#3396]) which approximates [`FunctionOptions.cast_options`]
  - Based on ([1], [2], [3]), this *could* just mean adding 1-3 members to `FunctionFlags`

[`plans::options::FunctionFlags`]: https://github.com/pola-rs/polars/blob/cd9de5da9d081acd33a5e886422062651ec6e2c4/crates/polars-plan/src/plans/options.rs#L54-L166
[`FunctionOptions`]: https://github.com/pola-rs/polars/blob/cd9de5da9d081acd33a5e886422062651ec6e2c4/crates/polars-plan/src/plans/options.rs#L183-L281
[`INPUT_WILDCARD_EXPANSION`]: https://github.com/pola-rs/polars/blob/b6ae11535a9a45a442446ad13f687616ca97ee95/crates/polars-plan/src/plans/options.rs#L66-L76
[#3396]: https://github.com/narwhals-dev/narwhals/pull/3396
[`FunctionOptions.cast_options`]: https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/plans/options.rs#L192-L195
[1]: https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/plans/options.rs#L263-L265
[2]: https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/plans/options.rs#L174
[3]: https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-core/src/utils/supertype.rs#L101-L138
"""

from __future__ import annotations

import enum
from typing import Any, Final

__all__ = ["FunctionFlags"]

_V_AGGREGATION: Final = 1 << 0
_V_LENGTH_PRESERVING: Final = 1 << 2
# NOTE: Declared externally, as `_missing_` raises on creation if the invariant is broken
_V_MUTUALLY_EXCLUSIVE: Final = _V_AGGREGATION | _V_LENGTH_PRESERVING


class FunctionFlags(enum.Flag):
    """Properties of functions.

    Flags tell us how a function transforms the shape of it's input:

        ┌───────────────────┬───────────────┬───────────┐
        │ Flag              ┆ Input         ┆ Output    │
        ╞═══════════════════╪═══════════════╪═══════════╡
        │ AGGREGATION       ┆ Column        ┆ Scalar    │
        │ ROW_SEPARABLE     ┆ Column        ┆ Unknown   │
        │ LENGTH_PRESERVING ┆ Column/Scalar ┆ Preserved │
        │ ELEMENTWISE       ┆ Column/Scalar ┆ Preserved │
        └───────────────────┴───────────────┴───────────┘

    And that's the main nugget we can use to answer the question:
    > Is this function valid *here*?

    ## Examples
    Canonical members are returned on iteration:
    >>> AGGREGATION, ROW_SEPARABLE, LENGTH_PRESERVING = FunctionFlags

    Members can be compared by identity:
    >>> AGGREGATION is FunctionFlags.AGGREGATION
    True
    >>> ROW_SEPARABLE is FunctionFlags.ROW_SEPARABLE
    True
    >>> LENGTH_PRESERVING is FunctionFlags.LENGTH_PRESERVING
    True

    Aliases can use containment for members they are composed of:
    >>> ELEMENTWISE = FunctionFlags.ELEMENTWISE
    >>> LENGTH_PRESERVING in ELEMENTWISE
    True
    >>> ROW_SEPARABLE in ELEMENTWISE
    True

    That relationship goes in a single direction:
    >>> ELEMENTWISE in LENGTH_PRESERVING
    False
    >>> ELEMENTWISE in ROW_SEPARABLE
    False

    Recomposing an alias still has the same identity:
    >>> (LENGTH_PRESERVING | ROW_SEPARABLE) is FunctionFlags.ELEMENTWISE
    True

    Not all aliases are valid:
    >>> LENGTH_PRESERVING | AGGREGATION
    Traceback (most recent call last):
    TypeError: A function cannot both return a scalar and preserve length, they are mutually exclusive.
    """

    DEFAULT = 0
    """No flags set.

    Takes on the identity of other flags when combined:
    >>> FunctionFlags.DEFAULT | FunctionFlags.AGGREGATION
    <FunctionFlags.AGGREGATION: 1>
    """

    AGGREGATION = _V_AGGREGATION
    """Always returns a scalar and supports broadcasting.

    Mutually exclusive with `LENGTH_PRESERVING`.

    ## Examples
    >>> import narwhals._plan as nw
    >>> def show(frame: nw.DataFrame) -> None:
    ...     print(frame.to_polars())
    >>> data = {"a": [1, None], "b": [1, 2]}
    >>> df = nw.DataFrame.from_dict(data, backend="pyarrow")
    >>> show(df)
    shape: (2, 2)
    ┌──────┬─────┐
    │ a    ┆ b   │
    │ ---  ┆ --- │
    │ i64  ┆ i64 │
    ╞══════╪═════╡
    │ 1    ┆ 1   │
    │ null ┆ 2   │
    └──────┴─────┘

    These expressions each use a different kind of function:
    >>> aggregation = nw.col("a").null_count().alias("c")
    >>> aggregation._ir.is_scalar()
    True
    >>> row_separable = nw.col("a").drop_nulls().alias("c")
    >>> row_separable._ir.is_scalar()
    False

    But in an isolated `select` context, that detail isn't always visible:
    >>> show(df.select(aggregation))
    shape: (1, 1)
    ┌─────┐
    │ c   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    └─────┘

    Since they return the same result *for this dataset*:
    >>> show(df.select(row_separable))
    shape: (1, 1)
    ┌─────┐
    │ c   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    └─────┘

    Aggregation functions support broadcasting, because we know they **always return a scalar**:
    >>> show(df.with_columns(aggregation))
    shape: (2, 3)
    ┌──────┬─────┬─────┐
    │ a    ┆ b   ┆ c   │
    │ ---  ┆ --- ┆ --- │
    │ i64  ┆ i64 ┆ i64 │
    ╞══════╪═════╪═════╡
    │ 1    ┆ 1   ┆ 1   │
    │ null ┆ 2   ┆ 1   │
    └──────┴─────┴─────┘

    Whereas row-separable functions do not support broadcasting:
    >>> show(df.with_columns(row_separable))
    Traceback (most recent call last):
    narwhals.exceptions.ShapeError: Series c, length 1 doesn't match the DataFrame height of 2...

    Because returning a single value is data-dependent:
    >>> show(df.select(nw.col("b").drop_nulls()))
    shape: (2, 1)
    ┌─────┐
    │ b   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    └─────┘
    """

    ROW_SEPARABLE = 1 << 1
    """Does not depend on the context of surrounding rows.

    Only `drop_nulls`, `drop_nans`.

    ## Important
    Row-separable functions are partially defined by what they are **not**:

        ELEMENTWISE = ROW_SEPARABLE | LENGTH_PRESERVING
        #             ✔️              ❌

    This definition means they change the length of input columns.

    But this property **does not** extend to elementwise.
    """

    LENGTH_PRESERVING = _V_LENGTH_PRESERVING
    """Does not change the length of input columns.

    Mutually exclusive with `AGGREGATION`.

    Includes `rolling_*`, `cum_*`, `shift`.

    ## Important
    Length-preserving functions are partially defined by what they are **not**:

        ELEMENTWISE = ROW_SEPARABLE | LENGTH_PRESERVING
        #             ❌              ✔️

    This definition means they depend on the context of surrounding rows.

    But this property **does not** extend to elementwise.
    """

    ELEMENTWISE = ROW_SEPARABLE | LENGTH_PRESERVING
    """Does not depend on the context of surrounding rows, nor change the length of input columns.

    They're the most common kind and can often be used freely in SQL.
    """

    def __str__(self) -> str:
        if (name := self.name) is None:
            # https://github.com/python/typeshed/blob/46b01bf323f3d7ee8764e844327c4010bade07c3/stdlib/enum.pyi#L260
            # https://docs.python.org/3/howto/enum.html#flag
            # Should be unreachable, since `DEFAULT = 0`
            msg = f"Expected all {type(self).__name__!r} to have a name, got: {self!r}, name=None"
            raise NotImplementedError(msg)
        return name.replace("|", " | ")

    @classmethod
    def _missing_(cls, value: Any) -> Any:
        # Matches `enum.Flag.__contains__`
        if (value & _V_MUTUALLY_EXCLUSIVE) == _V_MUTUALLY_EXCLUSIVE:
            msg = "A function cannot both return a scalar and preserve length, they are mutually exclusive."
            raise TypeError(msg)
        return super()._missing_(value)  # pragma: no cover

    def is_elementwise(self) -> bool:
        return FunctionFlags.ELEMENTWISE in self

    def changes_length(self) -> bool:
        return self in _CHANGES_LENGTH


# NOTE: Has to be exactly one of these, a set avoids this issue:
#     `DEFAULT | ROW_SEPARABLE -> ROW_SEPARABLE`
_CHANGES_LENGTH = frozenset((FunctionFlags.DEFAULT, FunctionFlags.ROW_SEPARABLE))
