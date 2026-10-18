from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = {
    "a": [1, 3, 2, 2],
    "b": [4, 4, 6, 5],
    "z": [7.0, 8.0, 9.0, 8.5],
    "ties_between_nulls": [None, 1.0, 1.0, None],
    "signed_between_nulls": [None, -3.0, 2.0, None],
}


@pytest.mark.parametrize(
    ("col", "expected_min", "expected_max"),
    [
        ("a", 0, 1),
        ("b", 0, 2),
        ("z", 0, 2),
        ("ties_between_nulls", 1, 1),
        ("signed_between_nulls", 1, 2),
    ],
)
def test_arg_min_max_series(
    constructor_eager: ConstructorEager,
    col: str,
    expected_min: int,
    expected_max: int,
    request: pytest.FixtureRequest,
) -> None:
    if "cudf" in str(constructor_eager):
        reason = "not implemented"
        request.applymarker(pytest.mark.xfail(reason=reason))
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    series = nw.maybe_set_index(series, index=[1, 0, 9, 4])  # type: ignore[arg-type]
    assert_equal_data({col: [series.arg_min()]}, {col: [expected_min]})
    assert_equal_data({col: [series.arg_max()]}, {col: [expected_max]})


def test_arg_min_max_series_no_valid_values(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    series = df["signed_between_nulls"]
    all_null = series.filter(series.is_null())
    empty = series.head(0)
    assert all_null.arg_min() is None
    assert all_null.arg_max() is None
    assert empty.arg_min() is None
    assert empty.arg_max() is None


@pytest.mark.parametrize("method", ["arg_min", "arg_max"])
def test_arg_extremum_preserves_comparison_errors(method: str) -> None:
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    native = pd.Series([np.array([1, 2]), np.array([3, 4])], dtype=object)
    series = nw.from_native(native, series_only=True)

    # pandas<2.1 raises `TypeError` for this dtype, newer pandas `ValueError`.
    msg = "truth value of an array|not allowed for this dtype"
    with pytest.raises((TypeError, ValueError), match=msg):
        getattr(series, method)()
