from __future__ import annotations

from inspect import getdoc
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals.stable.v2 as nw_v2

if TYPE_CHECKING:
    from collections.abc import Iterator


def remove_docstring_examples(doc: str) -> str:
    if "Examples:" in doc:
        return doc[: doc.find("Examples:")].rstrip()
    return doc.rstrip()


def test_stable_api_completeness() -> None:
    v2_api = nw_v2.__all__
    main_namespace_api = nw.__all__
    extra = set(v2_api).difference(main_namespace_api)
    assert not extra
    missing = set(main_namespace_api).difference(v2_api).difference({"stable"})
    assert not missing


def test_stable_api_docstrings() -> None:
    main_namespace_api = nw.__all__
    for item in main_namespace_api:
        if item in {"from_dict"}:
            # We keep `native_namespace` around in the main namespace
            # until at least hierarchical forecast make a release
            continue
        if (doc := getdoc(getattr(nw, item))) is None:
            continue
        v2_doc = getdoc(getattr(nw_v2, item))
        assert v2_doc is not None
        assert remove_docstring_examples(v2_doc) == remove_docstring_examples(doc), item


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

    df_v2 = nw_v2.from_native(pl.DataFrame())
    df = nw.from_native(pl.DataFrame())
    for method_name, doc in _iter_api_method_docs(df):
        doc_v2 = getdoc(getattr(df_v2, method_name))
        assert doc_v2
        assert remove_docstring_examples(doc_v2) == remove_docstring_examples(doc)


def test_lazyframe_docstrings() -> None:
    pytest.importorskip("polars")
    import polars as pl

    ldf_v2 = nw_v2.from_native(pl.LazyFrame())
    ldf = nw.from_native(pl.LazyFrame())
    performance_warning = {"schema", "columns"}
    deprecated = {"tail", "gather_every"}
    for method_name, doc in _iter_api_method_docs(ldf, *performance_warning, *deprecated):
        doc_v2 = getdoc(getattr(ldf_v2, method_name))
        assert doc_v2
        assert remove_docstring_examples(doc_v2) == remove_docstring_examples(doc)


def test_series_docstrings() -> None:
    pytest.importorskip("polars")
    import polars as pl

    ser_v2 = nw_v2.from_native(pl.Series(), series_only=True)
    ser = nw.from_native(pl.Series(), series_only=True)
    for method_name, doc in _iter_api_method_docs(ser):
        if method_name in "hist":
            # This is still very unstable in Polars so we don't have it in stable.v2 yet.
            continue
        doc_v2 = getdoc(getattr(ser_v2, method_name))
        assert doc_v2
        assert remove_docstring_examples(doc_v2) == remove_docstring_examples(doc)
