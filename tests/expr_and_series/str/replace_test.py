from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

replace_data = [
    ({"a": ["123abc", "abc456"]}, r"abc\b", "ABC", 1, False, {"a": ["123ABC", "abc456"]}),
    ({"a": ["abc abc", "abc456"]}, r"abc", "", 1, False, {"a": [" abc", "456"]}),
    ({"a": ["abc abc abc", "456abc"]}, r"abc", "", -1, False, {"a": ["  ", "456"]}),
    (
        {"a": ["Dollar $ign", "literal"]},
        r"$",
        "S",
        -1,
        True,
        {"a": ["Dollar Sign", "literal"]},
    ),
]

replace_all_data = [
    ({"a": ["123abc", "abc456"]}, r"abc\b", "ABC", False, {"a": ["123ABC", "abc456"]}),
    ({"a": ["abc abc", "abc456"]}, r"abc", "", False, {"a": [" ", "456"]}),
    ({"a": ["abc abc abc", "456abc"]}, r"abc", "", False, {"a": ["  ", "456"]}),
    (
        {"a": ["Dollar $ign", "literal"]},
        r"$",
        "S",
        True,
        {"a": ["Dollar Sign", "literal"]},
    ),
]

# Edge cases from differential testing: byte-vs-char offsets, match position
# vs match text, re-matching inside inserted text, and boundary `n` values.
replace_edge_data = [
    pytest.param({"a": ["ααα-x"]}, "-", "@", 1, True, {"a": ["ααα@x"]}, id="non_ascii"),  # noqa: RUF001
    pytest.param(
        {"a": ["héllo wörld"]},
        "wörld",
        "@",
        1,
        True,
        {"a": ["héllo @"]},
        id="non_ascii_word",
    ),
    pytest.param(
        {"a": ["abcx abc"]},
        r"abc\b",
        "Z",
        1,
        False,
        {"a": ["abcx Z"]},
        id="word_boundary",
    ),
    pytest.param(
        {"a": ["aa"]}, "a", "ab", 2, True, {"a": ["abab"]}, id="insert_replacement"
    ),
    pytest.param({"a": ["abc"]}, "abc", "Z", 0, False, {"a": ["abc"]}, id="n_zero"),
    pytest.param({"a": ["abc"]}, "", "Z", 1, True, {"a": ["Zabc"]}, id="empty_pattern"),
]

replace_data_multivalue = [
    (
        {"a": ["123abc", "abc456"], "b": ["ghi", "jkl"]},
        r"abc",
        "b",
        1,
        False,
        {"a": ["123ghi", "jkl456"]},
    ),
    (
        {"a": ["abc abc", "abc456"], "b": ["ghi", "jkl"]},
        r"abc",
        "b",
        1,
        False,
        {"a": ["ghi abc", "jkl456"]},
    ),
    (
        {"a": ["abc abc abc", "456abc"], "b": ["ghi", "jkl"]},
        r"abc",
        "b",
        -1,
        False,
        {"a": ["ghi ghi ghi", "456jkl"]},
    ),
    (
        {"a": ["Dollar $ign", "literal"], "b": ["ghi", "jkl"]},
        r"$",
        "b",
        -1,
        True,
        {"a": ["Dollar ghiign", "literal"]},
    ),
]

replace_all_data_multivalue = [
    (
        {"a": ["123abc", "abc456"], "b": ["ghi", "jkl"]},
        r"abc",
        "b",
        False,
        {"a": ["123ghi", "jkl456"]},
    ),
    (
        {"a": ["abc abc", "abc456"], "b": ["ghi", "jkl"]},
        r"abc",
        "b",
        False,
        {"a": ["ghi ghi", "jkl456"]},
    ),
    (
        {"a": ["Dollar $ign", "literal"], "b": ["ghi", "jkl"]},
        r"$",
        "b",
        True,
        {"a": ["Dollar ghiign", "literal"]},
    ),
]


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"), replace_data
)
def test_str_replace_series_scalar(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.replace(
        pattern=pattern, value=value, n=n, literal=literal
    )
    assert_equal_data({"a": result_series}, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "literal", "expected"), replace_all_data
)
def test_str_replace_all_series_scalar(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.replace_all(pattern=pattern, value=value, literal=literal)
    assert_equal_data({"a": result_series}, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"), replace_data
)
def test_str_replace_expr_scalar(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    if any(x in str(constructor) for x in ("pyspark", "duckdb", "ibis")):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor} only supports `replace_all`.",
                raises=NotImplementedError,
            )
        )
    df = nw.from_native(constructor(data))
    result_df = df.select(
        nw.col("a").str.replace(pattern=pattern, value=value, n=n, literal=literal)
    )
    assert_equal_data(result_df, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"), replace_edge_data
)
def test_str_replace_edge_series_scalar(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    *,
    literal: bool,
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result_series = df["a"].str.replace(
        pattern=pattern, value=value, n=n, literal=literal
    )
    assert_equal_data({"a": result_series}, expected)


replace_value_edge_cases = [
    pytest.param(
        "(b)",
        "[$1]",
        {"a": ["aa[b]"]},
        "no capture-group expansion",
        id="group_expansion",
    ),
    pytest.param(
        "b", "\\1", {"a": ["aa\\1"]}, "invalid replacement string", id="backslash_literal"
    ),
]


@pytest.mark.parametrize(
    ("pattern", "value", "expected", "reason"), replace_value_edge_cases
)
def test_str_replace_value_edge_series_scalar(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    pattern: str,
    value: str,
    expected: dict[str, list[str]],
    reason: str,
) -> None:
    # Special replacement strings follow polars; pandas-likes and pyarrow differ.
    # Old numpy-backed pandas treats a single-character pattern as a literal, so
    # the invalid replacement string never reaches `re.sub` and polars is matched.
    if (
        "backslash_literal" in request.node.callspec.id
        and "pandas_constructor" in str(constructor_eager)
        and PANDAS_VERSION < (2,)
    ):
        pytest.skip(reason="single-character pattern treated as literal")
    if any(x in str(constructor_eager) for x in ("pandas", "modin", "cudf", "pyarrow")):
        request.applymarker(pytest.mark.xfail(reason=reason))
    df = nw.from_native(constructor_eager({"a": ["aab"]}), eager_only=True)
    result_series = df["a"].str.replace(pattern=pattern, value=value, n=1, literal=False)
    assert_equal_data({"a": result_series}, expected)


@pytest.mark.parametrize(
    ("pattern", "value", "expected", "reason"), replace_value_edge_cases
)
def test_str_replace_value_edge_expr_scalar(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    pattern: str,
    value: str,
    expected: dict[str, list[str]],
    reason: str,
) -> None:
    if any(x in str(constructor) for x in ("pyspark", "duckdb", "ibis")):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor} only supports `replace_all`.",
                raises=NotImplementedError,
            )
        )
    # Old numpy-backed pandas treats a single-character pattern as a literal, so
    # the invalid replacement string never reaches `re.sub` and polars is matched.
    if (
        "backslash_literal" in request.node.callspec.id
        and "pandas_constructor" in str(constructor)
        and PANDAS_VERSION < (2,)
    ):
        pytest.skip(reason="single-character pattern treated as literal")
    if any(x in str(constructor) for x in ("pandas", "modin", "cudf", "pyarrow", "dask")):
        request.applymarker(pytest.mark.xfail(reason=reason))
    df = nw.from_native(constructor({"a": ["aab"]}))
    result_df = df.select(
        nw.col("a").str.replace(pattern=pattern, value=value, n=1, literal=False)
    )
    assert_equal_data(result_df, expected)


def test_str_replace_null_value_series(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    # A null replacement leaves the input unchanged, matching polars.
    if any(x in str(constructor_eager) for x in ("pandas", "modin", "cudf", "pyarrow")):
        request.applymarker(
            pytest.mark.xfail(reason="only str replacement values", raises=TypeError)
        )
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 37, 0):
        request.applymarker(pytest.mark.xfail(reason="old polars propagates null"))
    df = nw.from_native(
        constructor_eager({"a": ["abc", "def"], "b": ["X", None]}), eager_only=True
    )
    result_series = df["a"].str.replace(pattern="b", value=df["b"], n=1, literal=True)
    assert_equal_data({"a": result_series}, {"a": ["aXc", "def"]})


def test_str_replace_null_value_expr(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("pyspark", "duckdb", "ibis")):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor} only supports `replace_all`.",
                raises=NotImplementedError,
            )
        )
    if any(x in str(constructor) for x in ("pandas", "modin", "cudf", "pyarrow", "dask")):
        request.applymarker(
            pytest.mark.xfail(reason="only str replacement values", raises=TypeError)
        )
    if "polars" in str(constructor) and POLARS_VERSION < (1, 37, 0):
        request.applymarker(pytest.mark.xfail(reason="old polars propagates null"))
    df = nw.from_native(constructor({"a": ["abc", "def"], "b": ["X", None]}))
    result_df = df.select(
        nw.col("a").str.replace(pattern="b", value=nw.col("b"), n=1, literal=True)
    )
    assert_equal_data(result_df, {"a": ["aXc", "def"]})


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"), replace_edge_data
)
def test_str_replace_edge_expr_scalar(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    *,
    literal: bool,
    expected: dict[str, list[str]],
) -> None:
    if any(x in str(constructor) for x in ("pyspark", "duckdb", "ibis")):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor} only supports `replace_all`.",
                raises=NotImplementedError,
            )
        )
    df = nw.from_native(constructor(data))
    result_df = df.select(
        nw.col("a").str.replace(pattern=pattern, value=value, n=n, literal=literal)
    )
    assert_equal_data(result_df, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "literal", "expected"), replace_all_data
)
def test_str_replace_all_expr_scalar(
    constructor: Constructor,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("a").str.replace_all(pattern=pattern, value=value, literal=literal)
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"), replace_data_multivalue
)
def test_str_replace_series_multivalue(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
    request: pytest.FixtureRequest,
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    if any(x in str(constructor_eager) for x in ("pyarrow", "pandas", "modin", "cudf")):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor_eager} does not support multivalue replacement",
                raises=TypeError,
            )
        )

    result_series = df["a"].str.replace(
        pattern=pattern, value=df[value], n=n, literal=literal
    )
    assert_equal_data({"a": result_series}, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "literal", "expected"), replace_all_data_multivalue
)
def test_str_replace_all_series_multivalue(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
    request: pytest.FixtureRequest,
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow", "pandas", "modin", "cudf")):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor_eager} only supports `replace_all`.",
                raises=TypeError,
            )
        )

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result_series = df["a"].str.replace_all(
        pattern=pattern, value=df[value], literal=literal
    )
    assert_equal_data({"a": result_series}, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"), replace_data_multivalue
)
def test_str_replace_expr_multivalue(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    if any(x in str(constructor) for x in ("pyspark", "duckdb", "ibis")):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor} only supports `replace_all`.",
                raises=NotImplementedError,
            )
        )
    elif any(
        x in str(constructor) for x in ("dask", "pyarrow", "pandas", "modin", "cudf")
    ):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor} does not support multivalue replacement",
                raises=TypeError,
            )
        )

    df = nw.from_native(constructor(data))
    result_df = df.select(
        nw.col("a").str.replace(
            pattern=pattern, value=nw.col(value), n=n, literal=literal
        )
    )
    assert_equal_data(result_df, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "literal", "expected"), replace_all_data_multivalue
)
def test_str_replace_all_expr_multivalue(
    constructor: Constructor,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
    request: pytest.FixtureRequest,
) -> None:
    if any(x in str(constructor) for x in ("dask", "pyarrow", "pandas", "modin", "cudf")):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor} does not support multivalue replacement",
                raises=TypeError,
            )
        )

    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("a").str.replace_all(pattern=pattern, value=nw.col(value), literal=literal)
    )
    assert_equal_data(result, expected)


def test_str_replace_errors_series(constructor_eager: ConstructorEager) -> None:
    context: Any
    only_str_supported = pytest.raises(
        TypeError, match=r"only supports str replacement values"
    )
    multivalue_binary_n = pytest.raises(Exception, match=r"'n > 1' not yet supported")

    df = nw.from_native(constructor_eager({"a": ["abc", "def", "ab"]}))

    ## .str.replace
    # all eager backends support scalar replacement
    df["a"].str.replace("ab", "XYZ", n=1)
    df["a"].str.replace("ab", "XYZ", n=2)

    # pyarrow, pandas, modin and cudf do not support multivalue replacement
    context = nullcontext()
    if any(x in str(constructor_eager) for x in ("pyarrow", "pandas", "modin", "cudf")):
        context = only_str_supported

    with context:
        df["a"].str.replace("ab", df["a"])

    # no backends support multivalue AND n > 1; others error out on multivalue
    context = (
        multivalue_binary_n if "polars" in str(constructor_eager) else only_str_supported
    )
    with context:
        df["a"].str.replace("ab", df["a"], n=2)

    ## .str.replace_all; all eager backends support scalar replacement
    df["a"].str.replace_all("ab", "XYZ")

    # pyarrow, pandas, modin and cudf do not support multivalue replacement
    context = (
        only_str_supported
        if any(
            x in str(constructor_eager) for x in ("pyarrow", "pandas", "modin", "cudf")
        )
        else nullcontext()
    )
    with context:
        df["a"].str.replace_all("ab", df["a"])


def test_str_replace_errors_expr(constructor: Constructor) -> None:
    context: Any
    not_implemented = pytest.raises(NotImplementedError)
    only_str_supported = pytest.raises(
        TypeError, match=r"only supports str replacement values"
    )

    df = nw.from_native(constructor({"a": ["abc", "def", "ab"]}))

    ## .str.replace
    context = (
        not_implemented
        if any(x in str(constructor) for x in ("duckdb", "ibis", "pyspark"))
        else nullcontext()
    )
    with context:
        df.select(nw.col("a").str.replace("ab", "XYZ", n=1))

    ## .str.replace multivalue; some dont implement replace, others dont support multivalue
    context = nullcontext()
    if any(x in str(constructor) for x in ("duckdb", "ibis", "pyspark")):
        context = not_implemented
    elif any(
        x in str(constructor) for x in ("dask", "pyarrow", "pandas", "modin", "cudf")
    ):
        context = only_str_supported

    with context:
        df.select(nw.col("a").str.replace("ab", nw.col("a"), n=1))

    ## .str.replace_all; all backends support .str.replace_all with scalar replacement
    df.select(nw.col("a").str.replace_all("ab", "a"))

    ## .str.replace_all multivalue
    context = (
        only_str_supported
        if any(
            x in str(constructor) for x in ("dask", "pyarrow", "pandas", "modin", "cudf")
        )
        else nullcontext()
    )
    with context:
        df.select(nw.col("a").str.replace_all("ab", nw.col("a")))
