from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from inspect import getdoc
from typing import Any
from typing import Iterator

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import DUCKDB_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data


def remove_docstring_examples(doc: str) -> str:
    if "Examples:" in doc:
        return doc[: doc.find("Examples:")].rstrip()
    return doc.rstrip()


def test_renamed_taxicab_norm(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
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


def test_renamed_taxicab_norm_dataframe(constructor: Constructor) -> None:
    # Suppose we have `DataFrame._l1_norm` in `stable.v1`, but remove it
    # in the main namespace. Here, we check that it's still usable from
    # the stable api.
    def func(df_any: Any) -> Any:
        df = nw_v1.from_native(df_any)
        df = df._l1_norm()
        return df.to_native()

    result = nw_v1.from_native(func(constructor({"a": [1, 2, 3, -4, 5]})))
    expected = {"a": [15]}
    assert_equal_data(result, expected)


def test_renamed_taxicab_norm_dataframe_narwhalify(constructor: Constructor) -> None:
    # Suppose we have `DataFrame._l1_norm` in `stable.v1`, but remove it
    # in the main namespace. Here, we check that it's still usable from
    # the stable api when using `narwhalify`.
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
        if (doc := getdoc(getattr(nw, item))) is None:
            continue
        if item in {"from_native", "narwhalify"}:
            # `eager_or_interchange` param was removed from main namespace,
            # but is still present in v1 docstring.
            continue
        if item == "Enum":
            # In v1 this was Polars-only, after that pandas ordered categoricals
            # started to be mapped to it too, so the docstring changed.
            continue
        v1_doc = getdoc(getattr(nw_v1, item))
        assert v1_doc is not None
        assert remove_docstring_examples(v1_doc) == remove_docstring_examples(doc), item


def _iter_api_method_docs(obj: Any, *exclude: str) -> Iterator[tuple[str, str]]:
    for name in dir(obj):
        if (
            not name.startswith("_")
            and name not in exclude
            and (doc := getdoc(getattr(obj, name)))
        ):
            yield name, doc


def test_dataframe_docstrings() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df_v1 = nw_v1.from_native(pl.DataFrame())
    df = nw.from_native(pl.DataFrame())
    for method_name, doc in _iter_api_method_docs(df):
        doc_v1 = getdoc(getattr(df_v1, method_name))
        assert doc_v1
        assert remove_docstring_examples(doc_v1) == remove_docstring_examples(doc)


def test_lazyframe_docstrings() -> None:
    pytest.importorskip("polars")
    import polars as pl

    ldf_v1 = nw_v1.from_native(pl.LazyFrame())
    ldf = nw.from_native(pl.LazyFrame())
    performance_warning = {"schema", "columns"}
    deprecated = {"tail", "gather_every"}
    for method_name, doc in _iter_api_method_docs(ldf, *performance_warning, *deprecated):
        doc_v1 = getdoc(getattr(ldf_v1, method_name))
        assert doc_v1
        assert remove_docstring_examples(doc_v1) == remove_docstring_examples(doc)


def test_series_docstrings() -> None:
    pytest.importorskip("polars")
    import polars as pl

    ser_v1 = nw_v1.from_native(pl.Series(), series_only=True)
    ser = nw.from_native(pl.Series(), series_only=True)
    for method_name, doc in _iter_api_method_docs(ser):
        doc_v1 = getdoc(getattr(ser_v1, method_name))
        assert doc_v1
        assert remove_docstring_examples(doc_v1) == remove_docstring_examples(doc)


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
