from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import assert_equal_data, uses_pyarrow_backend

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": ["2020-01-01", "2020-01-02", None]}
expected = {"a": [date(2020, 1, 1), date(2020, 1, 2), None]}


def test_to_date_with_fmt_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "cudf" in str(constructor_eager) or (
        "pandas" in str(constructor_eager)
        and not uses_pyarrow_backend(constructor=constructor_eager)
    ):
        reason = "Date type is not supported"
        request.applymarker(pytest.mark.xfail(reason=reason))

    result = nw.from_native(constructor_eager(data), eager_only=True)["a"].str.to_date(
        format="%Y-%m-%d"
    )
    assert isinstance(result.dtype, nw.Date)
    assert_equal_data({"a": result}, expected)


def test_to_date_infer_fmt_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "cudf" in str(constructor_eager) or (
        "pandas" in str(constructor_eager)
        and not uses_pyarrow_backend(constructor=constructor_eager)
    ):
        reason = "Date type is not supported"
        request.applymarker(pytest.mark.xfail(reason=reason))

    result = nw.from_native(constructor_eager(data), eager_only=True)["a"].str.to_date()
    assert isinstance(result.dtype, nw.Date)
    assert_equal_data({"a": result}, expected)


def test_to_date_with_fmt_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "cudf" in str(constructor) or (
        "pandas" in str(constructor) and not uses_pyarrow_backend(constructor=constructor)
    ):
        reason = "Date type is not supported"
        request.applymarker(pytest.mark.xfail(reason=reason))

    if "dask" in str(constructor):
        reason = "not implemented"
        request.applymarker(pytest.mark.xfail(reason=reason))

    result = nw.from_native(constructor(data)).select(
        a=nw.col("a").str.to_date(format="%Y-%m-%d")
    )
    result_schema = result.collect_schema()
    assert isinstance(result_schema["a"], nw.Date)

    assert_equal_data(result, expected)


def test_to_date_infer_fmt_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "cudf" in str(constructor) or (
        "pandas" in str(constructor) and not uses_pyarrow_backend(constructor=constructor)
    ):
        reason = "Date type is not supported"
        request.applymarker(pytest.mark.xfail(reason=reason))
    if "ibis" in str(constructor):
        reason = "Cannot infer format"
        request.applymarker(pytest.mark.xfail(reason=reason))

    if "dask" in str(constructor):
        reason = "not implemented"
        request.applymarker(pytest.mark.xfail(reason=reason))

    result = nw.from_native(constructor(data)).select(a=nw.col("a").str.to_date())
    result_schema = result.collect_schema()
    assert isinstance(result_schema["a"], nw.Date)

    assert_equal_data(result, expected)
