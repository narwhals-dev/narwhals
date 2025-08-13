from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

from narwhals import (
    Array,
    Boolean,
    Categorical,
    List,
    String,
    Struct,
    from_native,
    new_series,
)
from narwhals.testing.asserts.utils import raise_assertion_error

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals.typing import IntoSeriesT, SeriesT

    CheckFn: TypeAlias = Callable[[SeriesT, SeriesT], None]


def assert_series_equal(
    left: IntoSeriesT,
    right: IntoSeriesT,
    *,
    check_dtypes: bool = True,
    check_names: bool = True,
    check_order: bool = True,
    check_exact: bool = False,
    rel_tol: float = 1e-05,
    abs_tol: float = 1e-08,
    categorical_as_str: bool = False,
) -> None:
    """Assert that the left and right Series are equal.

    Raises a detailed `AssertionError` if the Series differ.
    This function is intended for use in unit tests.

    Arguments:
        left: The first Series to compare.
        right: The second Series to compare.
        check_dtypes: Requires data types to match.
        check_names: Requires names to match.
        check_order: Requires elements to appear in the same order.
        check_exact: Requires float values to match exactly. If set to `False`, values are
            considered equal when within tolerance of each other (see `rel_tol` and
            `abs_tol`). Only affects columns with a Float data type.
        rel_tol: Relative tolerance for inexact checking, given as a fraction of the
            values in `right`.
        abs_tol: Absolute tolerance for inexact checking.
        categorical_as_str: Cast categorical columns to string before comparing.
            Enabling this helps compare columns that do not share the same string cache.
    """
    __tracebackhide__ = True

    left_ = from_native(left, series_only=True, pass_through=False)
    right_ = from_native(right, series_only=True, pass_through=False)

    _check_metadata(left_, right_, check_dtypes=check_dtypes, check_names=check_names)

    left_, right_ = _maybe_apply_preprocessing(
        left_, right_, categorical_as_str=categorical_as_str, check_order=check_order
    )

    l_vals, r_vals = _check_null_values(left_, right_)

    if check_exact or not left_.dtype.is_float():
        _check_exact_values(
            l_vals,
            r_vals,
            check_dtypes=check_dtypes,
            check_exact=check_exact,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            categorical_as_str=categorical_as_str,
        )
    else:
        _check_approximate_values(l_vals, r_vals, rel_tol=rel_tol, abs_tol=abs_tol)


def _check_metadata(
    left: SeriesT, right: SeriesT, *, check_dtypes: bool, check_names: bool
) -> None:
    """Check basic equality properties: implementation, length, dtype, and names."""
    if (l_impl := left.implementation) != (r_impl := right.implementation):
        raise_assertion_error("Series", "implementation mismatch", l_impl, r_impl)

    if (l_len := len(left)) != (r_len := len(right)):
        raise_assertion_error("Series", "length mismatch", l_len, r_len)

    if (l_dtype := left.dtype) != (r_dtype := right.dtype) and check_dtypes:
        raise_assertion_error("Series", "dtype mismatch", l_dtype, r_dtype)

    if (l_name := left.name) != (r_name := right.name) and check_names:
        raise_assertion_error("Series", "name mismatch", l_name, r_name)


def _maybe_apply_preprocessing(
    left: SeriesT, right: SeriesT, *, categorical_as_str: bool, check_order: bool
) -> tuple[SeriesT, SeriesT]:
    """Apply preprocessing transformations like categorical casting and sorting."""
    l_dtype = left.dtype

    # TODO(FBruzzesi): Add coverage
    if isinstance(l_dtype, Categorical) and categorical_as_str:  # pragma: no cover
        left, right = left.cast(String()), right.cast(String())

    if not check_order:
        if l_dtype.is_nested():
            msg = "`check_order=False` is not supported (yet) with nested data type."
            raise NotImplementedError(msg)
        left, right = left.sort(), right.sort()

    return left, right


def _check_null_values(left: SeriesT, right: SeriesT) -> tuple[SeriesT, SeriesT]:
    """Check null value consistency and return non-null values."""
    if (l_null_count := left.null_count()) != (r_null_count := right.null_count()) or (
        (l_null_mask := left.is_null()) != (r_null_mask := right.is_null())
    ).any():
        raise_assertion_error("Series", "null value mismatch", l_null_count, r_null_count)

    return left.filter(~l_null_mask), right.filter(~r_null_mask)


def _check_exact_values(
    left: SeriesT,
    right: SeriesT,
    *,
    check_dtypes: bool,
    check_exact: bool,
    rel_tol: float,
    abs_tol: float,
    categorical_as_str: bool,
) -> None:
    """Check exact value equality for various data types."""
    l_impl = left.implementation
    l_dtype, r_dtype = left.dtype, right.dtype
    if l_dtype.is_numeric():
        # For _all_ numeric dtypes, we can use `is_close` with 0-tolerances to handle
        # inf and nan values out of the box.
        is_not_equal_mask = ~left.is_close(right, rel_tol=0, abs_tol=0, nans_equal=True)
    elif isinstance(l_dtype, (Array, List)) and isinstance(r_dtype, (Array, List)):
        check_fn = partial(
            assert_series_equal,
            check_dtypes=check_dtypes,
            check_names=False,
            check_order=True,
            check_exact=check_exact,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            categorical_as_str=categorical_as_str,
        )
        _check_list_like(left, right, l_dtype, r_dtype, check_fn=check_fn)
        # If `_check_list_like` didn't raise, then every nested element is equal
        is_not_equal_mask = new_series("", [False], dtype=Boolean(), backend=l_impl)
    elif isinstance(l_dtype, Struct) and isinstance(r_dtype, Struct):
        check_fn = partial(
            assert_series_equal,
            check_dtypes=True,
            check_names=True,
            check_order=True,
            check_exact=check_exact,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            categorical_as_str=categorical_as_str,
        )
        _check_struct(left, right, l_dtype, r_dtype, check_fn=check_fn)
        # If `_check_struct` didn't raise, then every nested element is equal
        is_not_equal_mask = new_series("", [False], dtype=Boolean(), backend=l_impl)
    else:
        is_not_equal_mask = left != right

    if is_not_equal_mask.any():
        raise_assertion_error("Series", "exact value mismatch", left, right)


def _check_approximate_values(
    left: SeriesT, right: SeriesT, *, rel_tol: float, abs_tol: float
) -> None:
    """Check approximate value equality with tolerance."""
    is_not_close_mask = ~left.is_close(
        right, rel_tol=rel_tol, abs_tol=abs_tol, nans_equal=True
    )

    if is_not_close_mask.any():
        raise_assertion_error(
            "Series",
            "values not within tolerance",
            left.filter(is_not_close_mask),
            right.filter(is_not_close_mask),
        )


def _check_list_like(
    l_vals: SeriesT,
    r_vals: SeriesT,
    l_dtype: List | Array,
    r_dtype: List | Array,
    check_fn: CheckFn[SeriesT],
) -> None:
    # Check row by row after transforming each array/list into a new series.
    # Notice that order within the array/list must be the same, regardless of
    # `check_order` value at the top level.
    impl = l_vals.implementation
    try:
        for l_val, r_val in zip(l_vals, r_vals):
            check_fn(
                new_series(name="", values=l_val, dtype=l_dtype.inner, backend=impl),  # type: ignore[arg-type]
                new_series(name="", values=r_val, dtype=r_dtype.inner, backend=impl),  # type: ignore[arg-type]
            )
    except AssertionError:
        raise_assertion_error("Series", "nested value mismatch", l_vals, r_vals)


def _check_struct(
    l_vals: SeriesT,
    r_vals: SeriesT,
    l_dtype: Struct,
    r_dtype: Struct,
    check_fn: CheckFn[SeriesT],
) -> None:
    # Check field by field as a separate column.
    # Notice that for struct's polars raises if:
    #   * field names are different but values are equal
    #   * dtype differs, regardless of `check_dtypes=False`
    #   * order applies only at top level
    try:
        for l_field, r_field in zip(l_dtype.fields, r_dtype.fields):
            check_fn(l_vals.struct.field(l_field.name), r_vals.struct.field(r_field.name))
    except AssertionError:
        raise_assertion_error("Series", "exact value mismatch", l_vals, r_vals)
