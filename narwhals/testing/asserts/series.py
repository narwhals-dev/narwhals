from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals import Categorical, String, from_native, new_series
from narwhals.testing.asserts.utils import raise_assertion_error

if TYPE_CHECKING:
    from narwhals.typing import SeriesT


def assert_series_equal(  # noqa: C901, PLR0912
    left: SeriesT,
    right: SeriesT,
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
            considered equal when within tolerance of each other (see `rel_tol` and `abs_tol`).
            Only affects columns with a Float data type.
        rel_tol: Relative tolerance for inexact checking, given as a fraction of the values in
            `right`.
        abs_tol: Absolute tolerance for inexact checking.
        categorical_as_str: Cast categorical columns to string before comparing.
            Enabling this helps compare columns that do not share the same string cache.
    """
    __tracebackhide__ = True

    left_ = from_native(left, series_only=True, pass_through=False)
    right_ = from_native(right, series_only=True, pass_through=False)

    if (l_impl := left_.implementation) != (r_impl := right_.implementation):
        raise_assertion_error("Series", "implementation mismatch", l_impl, r_impl)

    if (l_len := len(left_)) != (r_len := len(right_)):
        raise_assertion_error("Series", "length mismatch", l_len, r_len)

    if (l_dtype := left_.dtype) != (r_dtype := right_.dtype) and check_dtypes:
        raise_assertion_error("Series", "dtype mismatch", l_dtype, r_dtype)

    if (l_name := left_.name) != (r_name := right_.name) and check_names:
        raise_assertion_error("Series", "name mismatch", l_name, r_name)

    # TODO(FBruzzesi): Add coverage
    if isinstance(l_dtype, Categorical) and categorical_as_str:  # pragma: no cover
        left_, right_ = left_.cast(String()), right_.cast(String())

    if not check_order:
        if l_dtype.is_nested():
            msg = "`check_order=False` is not supported (yet) with nested data type."
            raise NotImplementedError(msg)
        left_, right_ = left_.sort(), right_.sort()

    if ((l_null_mask := left_.is_null()) != (r_null_mask := right_.is_null())).any() or (
        l_null_count := left_.null_count()
    ) != (r_null_count := right_.null_count()):
        raise_assertion_error("Series", "null value mismatch", l_null_count, r_null_count)

    l_vals, r_vals = left_.filter(~l_null_mask), right_.filter(~r_null_mask)

    if check_exact or not l_dtype.is_numeric():
        if l_dtype.is_numeric():
            # For numeric dtypes, we can use `is_close` with 0-tolerances to handle inf
            # and nan values.
            is_not_equal_mask = ~left_.is_close(
                right_, rel_tol=0, abs_tol=0, nans_equal=True
            )
        elif l_dtype.is_nested():
            # TODO(FBruzzesi): Handle nested with inner numerical
            # 1. Polars raises: AssertionError: Series are different (nested value mismatch)
            # 2. check_exact and tolerances are considered within the inner type as well
            is_not_equal_mask = new_series(
                l_vals.name,
                (
                    any(x != y for x, y in zip(l_val, r_val))
                    for l_val, r_val in zip(l_vals, r_vals)
                ),
                backend=l_impl,
            )
        else:
            is_not_equal_mask = l_vals != r_vals

        if is_not_equal_mask.any():
            raise_assertion_error("Series", "exact value mismatch", l_vals, r_vals)

    else:
        is_not_close_mask = ~left_.is_close(
            right_, rel_tol=rel_tol, abs_tol=abs_tol, nans_equal=True
        )

        if is_not_close_mask.any():
            raise_assertion_error(
                "Series",
                "values not within tolerance",
                l_vals.filter(is_not_close_mask),
                r_vals.filter(is_not_close_mask),
            )
