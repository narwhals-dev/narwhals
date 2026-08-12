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


def _skip_unsupported_zfill(
    request: pytest.FixtureRequest, constructor: Constructor | ConstructorEager
) -> None:
    if uses_pyarrow_backend(constructor) and PANDAS_VERSION < (3,):
        reason = (
            "pandas with pyarrow backend doesn't support str.zfill, see "
            "https://github.com/pandas-dev/pandas/issues/61485"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))
    if "pandas" in str(constructor) and PANDAS_VERSION < (1, 5):
        pytest.skip(reason="different zfill behavior")
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 5):
        pytest.skip(
            reason="`TypeError: argument 'length': 'Expr' object cannot be interpreted "
            "as an integer` in `expr.str.slice(1, length)`"
        )


def test_str_zfill(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    _skip_unsupported_zfill(request, constructor)
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.zfill(3))
    assert_equal_data(result, expected)


def test_str_zfill_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    _skip_unsupported_zfill(request, constructor_eager)
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].str.zfill(3)
    assert_equal_data({"a": result}, expected)


def test_str_zfill_zero_width(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    # pyarrow special-cases width zero; run every backend to confirm the rest do not.
    _skip_unsupported_zfill(request, constructor)
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.zfill(0))
    assert_equal_data(result, data)


def test_str_zfill_zero_width_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    _skip_unsupported_zfill(request, constructor_eager)
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].str.zfill(0)
    assert_equal_data({"a": result}, data)


def test_str_zfill_negative_width_raises(constructor_eager: ConstructorEager) -> None:
    # Before, pandas returned the string unchanged, Polars raised its own
    # "conversion from `i128` to `u64` failed", and pyarrow crashed inside utf8_lpad.
    # The message is pinned rather than just the exception type because Polars already
    # raised InvalidOperationError, so a type-only check passes against unfixed code.
    s = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    msg = r"`width` must be non-negative but got -1"
    with pytest.raises(nw.exceptions.InvalidOperationError, match=msg):
        s.str.zfill(-1)


def test_str_zfill_negative_width_expr_raises(constructor: Constructor) -> None:
    # Separate from the series test: Series.str goes straight to the compliant series,
    # so this is the only one of the two that covers the lazy backends.
    df = nw.from_native(constructor(data))
    msg = r"`width` must be non-negative but got -1"
    with pytest.raises(nw.exceptions.InvalidOperationError, match=msg):
        df.select(nw.col("a").str.zfill(-1))
