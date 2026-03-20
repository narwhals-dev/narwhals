from __future__ import annotations

import enum


# TODO @dangotbanned: rename a loads of other things
# - would be nice to have `FunctionExpr.flags`, `Function.flags`
# TODO @dangotbanned: Explain `FunctionFlags` (class)
# NOTE: Few things might be redundant
# - `staticmethod`s: could just be the members instead?
# - `is_*` guards: pretty low usage: (2,4,0,1,1) and doesn't provide type checking
#   - can export members for short paths, but loses member docs
class FunctionFlags(enum.Flag):
    """Properties of functions."""

    # TODO @dangotbanned: Experiment with using `0` instead of `1`
    # https://docs.python.org/3/howto/enum.html#flag
    DEFAULT = 1 << 0
    """No flags set.

    This flag is compatible with all others, but in isolation it's defining
    trait is that it is not any other flag.

    ## History
    - Upstream this *was* named `ALLOW_GROUP_AWARE`, but removed in [#23690]
    - Left a vestigial `FunctionOptions.groupwise()`

    [#23690]: https://github.com/pola-rs/polars/pull/23690
    """

    REDUCE_EXPANSION = 1 << 2
    """Use different semantics when expanding selectors.

    Upstream this is named [`INPUT_WILDCARD_EXPANSION`].

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

    [`INPUT_WILDCARD_EXPANSION`]: https://github.com/pola-rs/polars/blob/b6ae11535a9a45a442446ad13f687616ca97ee95/crates/polars-plan/src/plans/options.rs#L66-L76
    """

    AGGREGATION = 1 << 3
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

    ROW_SEPARABLE = 1 << 6
    """Does not depend on the context of surrounding rows.

    Only `drop_nulls`, `drop_nans`.

    ## Important
    Row-separable functions are partially defined by what they are **not**:

        ELEMENTWISE = ROW_SEPARABLE | LENGTH_PRESERVING
        #             ✔️              ❌

    This definition means they change the length of input columns.

    But this property **does not** extend to elementwise.

    ## History
    ~~`*_range`~~ since [#26549](https://github.com/pola-rs/polars/pull/26549)
    """

    LENGTH_PRESERVING = 1 << 7
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
        """Special-case of `__or__` for inputs from `map_batches`."""
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
