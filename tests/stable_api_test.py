from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from typing import Any

import polars as pl
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import Constructor
from tests.utils import assert_equal_data


def remove_docstring_examples(doc: str) -> str:
    if "Examples:" in doc:
        return doc[: doc.find("Examples:")].rstrip()
    return doc.rstrip()


def test_renamed_taxicab_norm(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    # Suppose we need to rename `_l1_norm` to `_taxicab_norm`.
    # We need `narwhals.stable.v1` to stay stable. So, we
    # make the change in `narwhals`, and then add the new method
    # to the subclass of `Expr` in `narwhals.stable.v1`.
    # Here, we check that anyone who wrote code using the old
    # API will still be able to use it, without the main namespace
    # getting cluttered by the new name.

    df = nw.from_native(constructor({"a": [1, 2, 3, -4, 5]}))
    result = df.with_columns(b=nw.col("a")._taxicab_norm())
    expected = {"a": [1, 2, 3, -4, 5], "b": [15] * 5}
    assert_equal_data(result, expected)

    with pytest.raises(AttributeError):
        result = df.with_columns(b=nw.col("a")._l1_norm())  # type: ignore[attr-defined]

    df_v1 = nw_v1.from_native(constructor({"a": [1, 2, 3, -4, 5]}))
    # The newer `_taxicab_norm` can still work in the old API, no issue.
    # It's new, so it couldn't be backwards-incompatible.
    result_v1 = df_v1.with_columns(b=nw_v1.col("a")._taxicab_norm())
    expected = {"a": [1, 2, 3, -4, 5], "b": [15] * 5}
    assert_equal_data(result_v1, expected)

    # The older `_l1_norm` still works in the stable api
    result_v1 = df_v1.with_columns(b=nw_v1.col("a")._l1_norm())
    assert_equal_data(result_v1, expected)


def test_renamed_taxicab_norm_dataframe(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    # Suppose we have `DataFrame._l1_norm` in `stable.v1`, but remove it
    # in the main namespace. Here, we check that it's still usable from
    # the stable api.
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    def func(df_any: Any) -> Any:
        df = nw_v1.from_native(df_any)
        df = df._l1_norm()
        return df.to_native()

    result = nw_v1.from_native(func(constructor({"a": [1, 2, 3, -4, 5]})))
    expected = {"a": [15]}
    assert_equal_data(result, expected)


def test_renamed_taxicab_norm_dataframe_narwhalify(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    # Suppose we have `DataFrame._l1_norm` in `stable.v1`, but remove it
    # in the main namespace. Here, we check that it's still usable from
    # the stable api when using `narwhalify`.

    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    @nw_v1.narwhalify
    def func(df: Any) -> Any:
        return df._l1_norm()

    result = nw_v1.from_native(func(constructor({"a": [1, 2, 3, -4, 5]})))
    expected = {"a": [15]}
    assert_equal_data(result, expected)


def test_stable_api_completeness() -> None:
    v_1_api = nw_v1.__all__
    main_namespace_api = nw.__all__
    extra = set(v_1_api).difference(main_namespace_api)
    assert not extra
    missing = set(main_namespace_api).difference(v_1_api).difference({"stable"})
    assert not missing


def test_stable_api_docstrings() -> None:
    main_namespace_api = nw.__all__
    for item in main_namespace_api:
        if getattr(nw, item).__doc__ is None:
            continue
        if item in ("from_native", "narwhalify"):
            # `eager_or_interchange` param was removed from main namespace,
            # but is still present in v1 docstring.
            continue
        v1_doc = remove_docstring_examples(getattr(nw_v1, item).__doc__)
        nw_doc = remove_docstring_examples(getattr(nw, item).__doc__)
        assert v1_doc == nw_doc, item


def test_dataframe_docstrings() -> None:
    stable_df = nw_v1.from_native(pl.DataFrame())
    df = nw.from_native(pl.DataFrame())
    api = [i for i in df.__dir__() if not i.startswith("_")]
    for item in api:
        assert remove_docstring_examples(
            getattr(stable_df, item).__doc__.replace(
                "import narwhals.stable.v1 as nw", "import narwhals as nw"
            )
        ) == remove_docstring_examples(getattr(df, item).__doc__), item


def test_lazyframe_docstrings() -> None:
    stable_df = nw_v1.from_native(pl.LazyFrame())
    df = nw.from_native(pl.LazyFrame())
    api = [i for i in df.__dir__() if not i.startswith("_")]
    for item in api:
        if item in ("schema", "columns"):
            # to avoid performance warning
            continue
        if item in ("tail",):
            # deprecated
            continue
        assert remove_docstring_examples(
            getattr(stable_df, item).__doc__.replace(
                "import narwhals.stable.v1 as nw", "import narwhals as nw"
            )
        ) == remove_docstring_examples(getattr(df, item).__doc__)


def test_series_docstrings() -> None:
    stable_df = nw_v1.from_native(pl.Series(), series_only=True)
    df = nw.from_native(pl.Series(), series_only=True)
    api = [i for i in df.__dir__() if not i.startswith("_")]
    for item in api:
        if getattr(df, item).__doc__ is None:
            continue
        assert remove_docstring_examples(
            getattr(stable_df, item).__doc__.replace(
                "import narwhals.stable.v1 as nw", "import narwhals as nw"
            )
        ) == remove_docstring_examples(getattr(df, item).__doc__), item


def test_dtypes(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw_v1.from_native(
        constructor({"a": [1], "b": [datetime(2020, 1, 1)], "c": [timedelta(1)]})
    )
    dtype = df.collect_schema()["b"]
    assert dtype in {nw_v1.Datetime}
    assert isinstance(dtype, nw_v1.Datetime)
    dtype = df.collect_schema()["c"]
    assert dtype in {nw_v1.Duration}
    assert isinstance(dtype, nw_v1.Duration)
