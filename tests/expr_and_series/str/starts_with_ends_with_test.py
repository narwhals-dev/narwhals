from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

EXPR_UNSUPPORTED = ("dask", "pyarrow", "pandas", "modin", "cudf")
data = {"a": ["fdas", "edfas"], "prefix_suffix": ["fda", "fas"]}


@pytest.mark.parametrize(
    ("suffix", "expected"), [("das", [True, False]), (nw.lit("das"), [True, False])]
)
def test_ends_with(
    constructor: Constructor, suffix: str | nw.Expr, expected: list[bool]
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.ends_with(suffix))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("suffix", "expected"), [(nw.col("prefix_suffix"), [False, True])]
)
def test_ends_with_multi(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    suffix: nw.Expr,
    expected: list[bool],
) -> None:
    if any(x in str(constructor) for x in EXPR_UNSUPPORTED):
        request.applymarker(pytest.mark.xfail(reason="Not supported", raises=TypeError))
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.ends_with(suffix))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(("suffix", "expected"), [("das", [True, False])])
def test_ends_with_series(
    constructor_eager: ConstructorEager, suffix: str, expected: list[bool]
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].str.ends_with(suffix))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(("suffix", "expected"), [(data["prefix_suffix"], [False, True])])
def test_ends_with_series_multi(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    suffix: str | nw.Expr,
    expected: list[bool],
) -> None:
    if any(x in str(constructor_eager) for x in EXPR_UNSUPPORTED):
        request.applymarker(pytest.mark.xfail(reason="Not supported", raises=TypeError))

    df = nw.from_native(constructor_eager(data), eager_only=True)
    suffix_series = nw.from_native(constructor_eager({"a": suffix}), eager_only=True)["a"]
    result = df.select(df["a"].str.ends_with(suffix_series))

    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("prefix", "expected"), [("fda", [True, False]), (nw.lit("fda"), [True, False])]
)
def test_starts_with(
    constructor: Constructor, prefix: str | nw.Expr, expected: list[bool]
) -> None:
    df = nw.from_native(constructor(data)).lazy()
    result = df.select(nw.col("a").str.starts_with(prefix))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("prefix", "expected"), [(nw.col("prefix_suffix"), [True, False])]
)
def test_starts_with_multi(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    prefix: nw.Expr,
    expected: list[bool],
) -> None:
    if any(x in str(constructor) for x in EXPR_UNSUPPORTED):
        request.applymarker(pytest.mark.xfail(reason="Not supported", raises=TypeError))

    df = nw.from_native(constructor(data)).lazy()
    result = df.select(nw.col("a").str.starts_with(prefix))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(("prefix", "expected"), [("fda", [True, False])])
def test_starts_with_series(
    constructor_eager: ConstructorEager, prefix: str, expected: list[bool]
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].str.starts_with(prefix))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(("prefix", "expected"), [(data["prefix_suffix"], [True, False])])
def test_starts_with_series_multi(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    prefix: list[str],
    expected: list[bool],
) -> None:
    if any(x in str(constructor_eager) for x in EXPR_UNSUPPORTED):
        request.applymarker(pytest.mark.xfail(reason="Not supported", raises=TypeError))

    prefix_series = nw.from_native(constructor_eager({"a": prefix}), eager_only=True)["a"]
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].str.starts_with(prefix_series))
    assert_equal_data(result, {"a": expected})
