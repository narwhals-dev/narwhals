from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals.stable.v1 as nw_v1

if TYPE_CHECKING:
    from tests.interchange.conftest import MainInstances


def test_get_level() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = pl.DataFrame({"a": [1, 2, 3]})
    assert nw_v1.get_level(nw_v1.from_native(df)) == "full"
    assert (
        nw_v1.get_level(
            nw_v1.from_native(df.__dataframe__(), eager_or_interchange_only=True)
        )
        == "interchange"
    )


def test_v1_explicit_level_kwarg() -> None:
    pytest.importorskip("polars")
    import polars as pl

    nw_lf = nw_v1.from_native(pl.LazyFrame({"a": [1]}))
    rewrapped_lf = nw_v1.LazyFrame[pl.LazyFrame](nw_lf._compliant_frame)
    assert nw_v1.get_level(rewrapped_lf) == "lazy"

    nw_s = nw_v1.from_native(pl.Series(name="a", values=[1]), series_only=True)
    rewrapped_s = nw_v1.Series(nw_s._compliant_series)
    assert nw_v1.get_level(rewrapped_s) == "full"


def test_get_level_raises_main(main_instances: MainInstances) -> None:
    df, lf, ser = main_instances
    with pytest.raises(TypeError):
        nw_v1.get_level(df)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        nw_v1.get_level(ser)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        nw_v1.get_level(lf)  # type: ignore[arg-type]


def test_get_level_main_via_v1_from_native(main_instances: MainInstances) -> None:
    df, lf, ser = main_instances
    df_v1 = nw_v1.from_native(df)
    # NOTE: Typing doesn't match the behavior following (#3515)
    _lf_v1 = nw_v1.from_native(lf)  # type: ignore[call-overload]
    lf_v1: nw_v1.LazyFrame[Any] = _lf_v1
    ser_v1 = nw_v1.from_native(ser, series_only=True)

    assert nw_v1.get_level(df_v1) == "full"
    assert nw_v1.get_level(ser_v1) == "full"
    if lf_v1.implementation.is_polars():
        assert nw_v1.get_level(lf_v1) == "lazy"
    else:
        # Well this is strange?
        assert nw_v1.get_level(lf_v1) == "full"
