from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    uses_pyarrow_backend,
)

data = {"a": ["-1", "+1", "1", "12", "123", "99999", "+9999", None]}
expected = {"a": ["-01", "+01", "001", "012", "123", "99999", "+9999", None]}


def test_str_zfill(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if uses_pyarrow_backend(constructor) and PANDAS_VERSION < (3,):
        reason = (
            "pandas with pyarrow backend doesn't support str.zfill, see "
            "https://github.com/pandas-dev/pandas/issues/61485"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))

    if "pandas" in str(constructor) and PANDAS_VERSION < (1, 5):
        reason = "different zfill behavior"
        pytest.skip(reason=reason)

    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 5):
        reason = (
            "`TypeError: argument 'length': 'Expr' object cannot be interpreted as an integer`"
            "in `expr.str.slice(1, length)`"
        )
        pytest.skip(reason=reason)

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.zfill(3))
    assert_equal_data(result, expected)


def test_str_zfill_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if uses_pyarrow_backend(constructor_eager) and PANDAS_VERSION < (3,):
        reason = (
            "pandas with pyarrow backend doesn't support str.zfill, see "
            "https://github.com/pandas-dev/pandas/issues/61485"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))

    if "pandas" in str(constructor_eager) and PANDAS_VERSION < (1, 5):
        reason = "different zfill behavior"
        pytest.skip(reason=reason)

    if "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 5):
        reason = (
            "`TypeError: argument 'length': 'Expr' object cannot be interpreted as an integer`"
            "in `expr.str.slice(1, length)`"
        )
        pytest.skip(reason=reason)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].str.zfill(3)
    assert_equal_data({"a": result}, expected)


def test_str_zfill_zero_width_pyarrow() -> None:
    # A zero width is a no-op (every string is already at least 0 long). This used
    # to crash on the pyarrow backend with `ArrowInvalid: Negative buffer resize`,
    # because pc.case_when eagerly evaluated the utf8_lpad(width - 1) = -1 branch.
    pytest.importorskip("pyarrow")
    result = nw.from_dict(data, backend="pyarrow")["a"].str.zfill(0)
    assert_equal_data({"a": result}, data)


def test_str_zfill_negative_width_raises(constructor_eager: ConstructorEager) -> None:
    # A negative width is rejected in the public layer, so every backend gives the same
    # error. Before, pandas returned the string unchanged, Polars surfaced an internal
    # "conversion from `i128` to `u64` failed", and pyarrow crashed inside utf8_lpad.
    s = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    msg = r"`width` must be non-negative but got -1"
    with pytest.raises(nw.exceptions.InvalidOperationError, match=msg):
        s.str.zfill(-1)


def test_str_zfill_negative_width_expr_raises(constructor: Constructor) -> None:
    # Separate from the series test so the lazy backends are covered too: the check
    # lives in the expression layer, so it raises at select() rather than at collect().
    df = nw.from_native(constructor(data))
    msg = r"`width` must be non-negative but got -1"
    with pytest.raises(nw.exceptions.InvalidOperationError, match=msg):
        df.select(nw.col("a").str.zfill(-1))
