from typing import Any

import polars as pl
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import compare_dicts


def test_renamed_taxicab_norm(constructor: Any) -> None:
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
    compare_dicts(result, expected)

    with pytest.raises(AttributeError):
        result = df.with_columns(b=nw.col("a")._l1_norm())  # type: ignore[attr-defined]

    df = nw_v1.from_native(constructor({"a": [1, 2, 3, -4, 5]}))
    # The newer `_taxicab_norm` can still work in the old API, no issue.
    # It's new, so it couldn't be backwards-incompatible.
    result = df.with_columns(b=nw_v1.col("a")._taxicab_norm())
    expected = {"a": [1, 2, 3, -4, 5], "b": [15] * 5}
    compare_dicts(result, expected)

    # The older `_l1_norm` still works in the stable api
    result = df.with_columns(b=nw_v1.col("a")._l1_norm())
    compare_dicts(result, expected)


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
        assert (
            getattr(nw_v1, item).__doc__.replace(
                "import narwhals.stable.v1 as nw", "import narwhals as nw"
            )
            == getattr(nw, item).__doc__
        )
        assert (
            getattr(nw, item).__doc__.replace(
                "import narwhals as nw", "import narwhals.stable.v1 as nw"
            )
            == getattr(nw_v1, item).__doc__
        )


def test_dataframe_docstrings() -> None:
    stable_df = nw_v1.from_native(pl.DataFrame())
    df = nw.from_native(pl.DataFrame())
    api = [i for i in df.__dir__() if not i.startswith("_")]
    for item in api:
        assert (
            getattr(stable_df, item).__doc__.replace(
                "import narwhals.stable.v1 as nw", "import narwhals as nw"
            )
            == getattr(df, item).__doc__
        )


def test_lazyframe_docstrings() -> None:
    stable_df = nw_v1.from_native(pl.LazyFrame())
    df = nw.from_native(pl.LazyFrame())
    api = [i for i in df.__dir__() if not i.startswith("_")]
    for item in api:
        if item in ("schema", "columns"):
            # to avoid performance warning
            continue
        assert (
            getattr(stable_df, item).__doc__.replace(
                "import narwhals.stable.v1 as nw", "import narwhals as nw"
            )
            == getattr(df, item).__doc__
        )


def test_series_docstrings() -> None:
    stable_df = nw_v1.from_native(pl.Series(), series_only=True)
    df = nw.from_native(pl.Series(), series_only=True)
    api = [i for i in df.__dir__() if not i.startswith("_")]
    for item in api:
        if getattr(df, item).__doc__ is None:
            continue
        assert (
            getattr(stable_df, item).__doc__.replace(
                "import narwhals.stable.v1 as nw", "import narwhals as nw"
            )
            == getattr(df, item).__doc__
        )
