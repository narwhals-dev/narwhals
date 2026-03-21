"""Home to `FunctionFlags`.

## Implementation Notes
- Adapted from [`plans::options::FunctionFlags`] and folds the outer struct [`FunctionOptions`] into a single enum
- `REDUCE_EXPANSION` was renamed from [`INPUT_WILDCARD_EXPANSION`]
  - *Wildcard* references an old name for `col('*')`
  - *Reduce* references what `*_horizontal` functions do
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

_V_AGGREGATION: Final = 1 << 1
_V_LENGTH_PRESERVING: Final = 1 << 3
# NOTE: Declared externally, as `_missing_` raises on creation if the invariant is broken
_V_MUTUALLY_EXCLUSIVE: Final = _V_AGGREGATION | _V_LENGTH_PRESERVING


# TODO @dangotbanned: Explain `FunctionFlags` (class)
class FunctionFlags(enum.Flag):
    """Behaviors of a function."""

    DEFAULT = 0
    """No flags set.

    This flag is compatible with all others, but in isolation it's defining
    trait is that it is not any other flag.
    """

    REDUCE_EXPANSION = 1 << 0
    """Use different semantics when expanding selectors.

    ## Examples
    Say we have the following schema:
    >>> from tests.plan.utils import Frame
    >>> import narwhals._plan as nw

    >>> df = Frame.from_names("a", "b", "c")
    >>> df.schema
    Schema({'a': Int64, 'b': Int64, 'c': Int64})

    This kind of expansion is used for inputs to horizontal functions:
    >>> before = nw.sum_horizontal(nw.all())
    >>> (reduced,) = df.project(before)
    >>> before._ir
    ncs.all().sum_horizontal()
    >>> reduced
    a=col('a').sum_horizontal([col('b'), col('c')])

    Whereas the more common form of expansion produces multiple outputs:
    >>> before = nw.all().clip("b")
    >>> before._ir
    ncs.all().clip_lower([col('b')])
    >>> df.project(before)  # doctest: +NORMALIZE_WHITESPACE
    (a=col('a').clip_lower([col('b')]),
     b=col('b').clip_lower([col('b')]),
     c=col('c').clip_lower([col('b')]))
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

    # TODO @dangotbanned: Remove from `*_range`
    # https://github.com/pola-rs/polars/pull/26549
    ROW_SEPARABLE = 1 << 2
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
    HORIZONTAL = REDUCE_EXPANSION | ELEMENTWISE

    def __str__(self) -> str:
        # NOTE: Appeared as `self.flags` in:
        #   `f"{type(self).__name__}(flags='{self.flags}')"``
        name = self.name or "<FUNCTION_FLAGS_UNKNOWN>"
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
