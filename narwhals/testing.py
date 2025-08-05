from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

from narwhals.functions import new_series

if TYPE_CHECKING:
    from narwhals.typing import SeriesT


def _raise_assertion_error(mismatch_type: str, left: Any, right: Any) -> NoReturn:
    msg = f"Series are different ({mismatch_type})\n[left]: {left}\n[right]: {right}"

    raise AssertionError(msg)


def assert_series_equal(
    left: SeriesT,
    right: SeriesT,
    *,
    check_backend: bool = True,
    check_dtypes: bool = True,
    check_names: bool = True,
    # check_order: bool = True,  # Don't allow False due to complex types
    check_exact: bool = False,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    # categorical_as_str: bool = False,  # TODO(FBruzzesi): Should we even support this?
) -> None:
    if check_backend and (l_impl := left.implementation) != (
        r_impl := right.implementation
    ):
        _raise_assertion_error("backend mismatch", l_impl, r_impl)

    if (l_len := len(left)) != (r_len := len(right)):
        _raise_assertion_error("length mismatch", l_len, r_len)

    if check_dtypes and (l_dtype := left.dtype) != (r_dtype := right.dtype):
        _raise_assertion_error("dtype mismatch", l_dtype, r_dtype)

    if check_names and (l_name := left.name) != (r_name := right.name):
        _raise_assertion_error("name mismatch", l_name, r_name)

    if ((l_null_mask := left.is_null()) != (r_null_mask := right.is_null())).any() or (
        l_null_count := left.null_count()
    ) != (r_null_count := right.null_count()):
        _raise_assertion_error("null value mismatch", l_null_count, r_null_count)

    l_vals, r_vals = left.filter(~l_null_mask), right.filter(~r_null_mask)

    # TODO(FBruzzesi): Open points:
    #  1. Handle nan's and infinities for numerical
    #  2. Nested values: do nested values compare "nicely"?
    #  3. date vs datetime
    if check_exact or not l_dtype.is_numeric():
        if not (l_vals == r_vals).all():
            _raise_assertion_error("exact value mismatch", l_vals, r_vals)

    else:
        # Based on math.isclosed https://docs.python.org/3/library/math.html#math.isclose
        # Namely: abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol).
        l_abs, r_abs = l_vals.abs(), r_vals.abs()
        m1 = l_abs.zip_with(l_abs > r_abs, r_abs) * rtol
        m2 = new_series(name="tmp", values=[atol] * l_len, backend=l_impl)

        is_not_close_mask = (l_vals - r_vals).abs() > m1.zip_with(m1 > m2, m2)

        if is_not_close_mask.any():
            _raise_assertion_error(
                "values not within tolerance",
                l_vals.filter(is_not_close_mask),
                r_vals.filter(is_not_close_mask),
            )
