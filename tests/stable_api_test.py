from __future__ import annotations

from inspect import getdoc
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1

if TYPE_CHECKING:
    from collections.abc import Iterator


def remove_docstring_examples(doc: str) -> str:
    if "Examples:" in doc:
        return doc[: doc.find("Examples:")].rstrip()
    return doc.rstrip()


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
        if item in {"from_native", "narwhalify", "get_level"}:
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
