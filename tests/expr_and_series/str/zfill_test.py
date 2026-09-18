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

zfill_cases = [
    pytest.param(
        {"a": ["-1", "+1", "1", "12", "123", "99999", "+9999", None]},
        3,
        {"a": ["-01", "+01", "001", "012", "123", "99999", "+9999", None]},
        id="basic",
    ),
    pytest.param({"a": ["-", "+", ""]}, 3, {"a": ["-00", "+00", "000"]}, id="sign_only"),
    pytest.param(
        {"a": ["-1", "+1", "1", "", None]},
        1,
        {"a": ["-1", "+1", "1", "0", None]},
        id="width_1",
    ),
    pytest.param(
        {"a": ["-1", "+1", "1", "", None]},
        0,
        {"a": ["-1", "+1", "1", "", None]},
        id="width_0",
    ),
    pytest.param({"a": ["日本", None]}, 3, {"a": ["日本", None]}, id="non_ascii"),
    pytest.param({"a": ["+é", "+12"]}, 3, {"a": ["+é", "+12"]}, id="multibyte_plus_w3"),
    pytest.param({"a": ["+é", "+12"]}, 4, {"a": ["+0é", "+012"]}, id="multibyte_plus_w4"),
    pytest.param(
        {"a": ["+é", "+12"]}, 5, {"a": ["+00é", "+0012"]}, id="multibyte_plus_w5"
    ),
]


@pytest.mark.parametrize(("data", "width", "expected"), zfill_cases)
def test_str_zfill(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    data: dict[str, list[str | None]],
    width: int,
    expected: dict[str, list[str | None]],
) -> None:
    # Width 0 short-circuits before reaching the native implementation.
    if (
        "width_0" not in request.node.callspec.id
        and uses_pyarrow_backend(constructor)
        and PANDAS_VERSION < (3,)
    ):
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

    if any(
        case_id in request.node.callspec.id for case_id in ("non_ascii", "multibyte_plus")
    ) and "polars" not in str(constructor):
        request.applymarker(
            pytest.mark.xfail(reason="non-polars backends count characters")
        )

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.zfill(width))
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("data", "width", "expected"), zfill_cases)
def test_str_zfill_series(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    data: dict[str, list[str | None]],
    width: int,
    expected: dict[str, list[str | None]],
) -> None:
    # Width 0 short-circuits before reaching the native implementation.
    if (
        "width_0" not in request.node.callspec.id
        and uses_pyarrow_backend(constructor_eager)
        and PANDAS_VERSION < (3,)
    ):
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

    if any(
        case_id in request.node.callspec.id for case_id in ("non_ascii", "multibyte_plus")
    ) and "polars" not in str(constructor_eager):
        request.applymarker(
            pytest.mark.xfail(reason="non-polars backends count characters")
        )

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].str.zfill(width)
    assert_equal_data({"a": result}, expected)


def test_zfill_negative_width_expr(constructor: Constructor) -> None:
    # NOTE: The expression raises while it is built, so `select` is never reached.
    # The frame here allows to keep the test in case validation moves back into the backends.
    df = nw.from_native(constructor({"a": ["foo", None]}))
    with pytest.raises(ValueError, match="non-negative `width`"):
        df.select(nw.col("a").str.zfill(-1))


def test_zfill_negative_width_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager({"a": ["foo", None]}), eager_only=True)["a"]
    with pytest.raises(ValueError, match="non-negative `width`"):
        series.str.zfill(-1)
