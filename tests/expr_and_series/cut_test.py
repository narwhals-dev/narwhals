from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
import narwhals.stable.v2 as nw_v2
from narwhals.exceptions import ComputeError, DuplicateError, ShapeError
from tests.utils import Constructor, ConstructorEager, assert_equal_data

inf = float("inf")
data = {"a": [-inf, -1.0, 0.0, 0.5, 2.0, 3.0, inf, None]}


def xfail_if_not_implemented(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(
        x in str(constructor) for x in ("dask", "duckdb", "ibis", "pyspark", "sqlframe")
    ):
        reason = "`cut` is not implemented yet for Dask and the SQL backends"
        request.applymarker(pytest.mark.xfail(reason=reason, raises=NotImplementedError))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {},
            [
                "(-inf, 0]",
                "(-inf, 0]",
                "(-inf, 0]",
                "(0, 2]",
                "(0, 2]",
                "(2, inf]",
                "(2, inf]",
                None,
            ],
        ),
        (
            {"left_closed": True},
            [
                "[-inf, 0)",
                "[-inf, 0)",
                "[0, 2)",
                "[0, 2)",
                "[2, inf)",
                "[2, inf)",
                "[2, inf)",
                None,
            ],
        ),
        (
            {"labels": ["low", "mid", "high"]},
            ["low", "low", "low", "mid", "mid", "high", "high", None],
        ),
    ],
)
def test_cut_expr(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    kwargs: dict[str, Any],
    expected: list[str | None],
) -> None:
    xfail_if_not_implemented(constructor, request)
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").cut([0, 2], **kwargs))
    assert_equal_data(result, {"a": expected})


def test_cut_expr_int(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    xfail_if_not_implemented(constructor, request)
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(nw.col("a").cut([2]))
    assert_equal_data(result, {"a": ["(-inf, 2]", "(-inf, 2]", "(2, inf]"]})


@pytest.mark.parametrize(
    ("breaks", "values", "expected"),
    [
        ([0.5, 2.0], [0.0, 1.0, 3.0], ["(-inf, 0.5]", "(0.5, 2]", "(2, inf]"]),
        ([-3, -1.5], [-4.0, -2.0, 0.0], ["(-inf, -3]", "(-3, -1.5]", "(-1.5, inf]"]),
        (
            [1, 1e20],
            [0.0, 2.0, 1e21],
            ["(-inf, 1]", "(1, 100000000000000000000]", "(100000000000000000000, inf]"],
        ),
        ([1e-7], [0.0, 1.0], ["(-inf, 0.0000001]", "(0.0000001, inf]"]),
        ([], [1.0], ["(-inf, inf]"]),
    ],
)
def test_cut_series_default_labels(
    constructor_eager: ConstructorEager,
    breaks: list[float],
    values: list[float],
    expected: list[str],
) -> None:
    series = nw.from_native(constructor_eager({"a": values}), eager_only=True)["a"]
    result = series.cut(breaks)
    assert_equal_data({"a": result}, {"a": expected})


def test_cut_series_nan_is_null(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(
        constructor_eager({"a": [float("nan"), 1.0]}), eager_only=True
    )["a"]
    result = series.cut([0])
    assert_equal_data({"a": result}, {"a": [None, "(0, inf]"]})


def test_cut_series_dtype(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager({"a": [1.0, 3.0]}), eager_only=True)["a"]
    result = series.cut([2], labels=["lo", "hi"])
    if constructor_eager.__name__ == "pyarrow_table_constructor":
        # Narwhals has no `Enum` type for PyArrow, so the bins come back as `Categorical`.
        assert result.dtype == nw.Categorical
    else:
        assert result.dtype == nw.Enum(["lo", "hi"])


@pytest.mark.parametrize(
    ("breaks", "labels", "exception", "match"),
    [
        ([1, 1], None, DuplicateError, "breaks are not unique"),
        ([2, 1], None, ComputeError, "breaks must increase monotonically"),
        ([float("nan")], None, ComputeError, "breaks must not contain NaN or infinity"),
        ([0, inf], None, ComputeError, "breaks must not contain NaN or infinity"),
        ([1], ["a"], ShapeError, r"expected 2 labels \(len\(breaks\) \+ 1\), got 1"),
        ([1], ["a", "a"], DuplicateError, "labels are not unique"),
    ],
)
def test_cut_invalid_arguments(
    breaks: list[float], labels: list[str] | None, exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        nw.col("a").cut(breaks, labels=labels)


@pytest.mark.parametrize("stable", [nw_v1, nw_v2])
def test_cut_not_in_stable(stable: Any) -> None:
    with pytest.raises(NotImplementedError, match="unstable"):
        stable.col("a").cut([0])
    pa = pytest.importorskip("pyarrow")
    series = stable.from_native(pa.chunked_array([[1.0]]), series_only=True)
    with pytest.raises(NotImplementedError, match="unstable"):
        series.cut([0])
