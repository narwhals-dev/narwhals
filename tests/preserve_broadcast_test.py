from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from narwhals._compliant import EagerSeries

    SeriesFactory: TypeAlias = "Callable[[list[Any]], Any]"
    Case: TypeAlias = "tuple[str, list[Any], Callable[[Any], EagerSeries[Any]]]"


def pandas_series(data: list[Any]) -> Any:
    import pandas as pd

    from narwhals._pandas_like.series import PandasLikeSeries

    return PandasLikeSeries(
        pd.Series(data), implementation=Implementation.PANDAS, version=Version.MAIN
    )


def arrow_series(data: list[Any]) -> Any:
    import pyarrow as pa

    from narwhals._arrow.series import ArrowSeries

    return ArrowSeries(pa.chunked_array([data]), name="", version=Version.MAIN)


def like(series: Any, data: list[Any]) -> Any:
    """Build an operand of the same backend and length as `series`."""
    return series.from_iterable(data * len(series), context=series)


# Elementwise and length-preserving: the flag must survive.
PRESERVING: list[Case] = [
    ("__invert__", [True], lambda s: ~s),
    ("__neg__", [1], lambda s: -s),
    ("abs", [-1], lambda s: s.abs()),
    ("ceil", [1.2], lambda s: s.ceil()),
    ("clip", [5], lambda s: s.clip(like(s, [1]), like(s, [9]))),
    ("clip_lower", [5], lambda s: s.clip_lower(like(s, [1]))),
    ("clip_upper", [5], lambda s: s.clip_upper(like(s, [9]))),
    ("cos", [1.0], lambda s: s.cos()),
    ("exp", [1.0], lambda s: s.exp()),
    ("floor", [1.2], lambda s: s.floor()),
    ("is_between", [5], lambda s: s.is_between(like(s, [1]), like(s, [9]), "both")),
    ("is_finite", [1.0], lambda s: s.is_finite()),
    ("is_in", [1], lambda s: s.is_in([1, 2])),
    ("log", [8.0], lambda s: s.log(2.0)),
    (
        "replace_strict",
        [1],
        lambda s: s.replace_strict(like(s, [0]), [1], [2], return_dtype=None),
    ),
    ("round", [1.234], lambda s: s.round(2)),
    ("sin", [1.0], lambda s: s.sin()),
    ("sqrt", [4.0], lambda s: s.sqrt()),
    ("zip_with", ["a"], lambda s: s.zip_with(like(s, [True]), like(s, ["b"]))),
]

# Leaving length 1 must drop the flag, whatever the operation.
LOSES_LENGTH_1: list[Case] = [
    ("filter", [1], lambda s: s.filter(like(s, [False]))),
    ("drop_nulls", [None], lambda s: s.drop_nulls()),
    ("head", [1], lambda s: s.head(0)),
    (
        "_align_full_broadcast",
        [1],
        lambda s: type(s)._align_full_broadcast(s, like(s, [1, 2, 3]))[0],
    ),
]


@pytest.fixture(params=["pandas", "pyarrow"])
def factory(request: pytest.FixtureRequest) -> SeriesFactory:
    pytest.importorskip(request.param)
    return pandas_series if request.param == "pandas" else arrow_series


@pytest.mark.parametrize(
    ("name", "data", "call"), PRESERVING, ids=[case[0] for case in PRESERVING]
)
def test_preserves_broadcast(
    factory: SeriesFactory,
    name: str,
    data: list[Any],
    call: Callable[[Any], EagerSeries[Any]],
) -> None:
    series = factory(data)
    series._broadcast = True
    assert call(series)._broadcast is True, name


@pytest.mark.parametrize(
    ("name", "data", "call"), LOSES_LENGTH_1, ids=[case[0] for case in LOSES_LENGTH_1]
)
def test_drops_broadcast(
    factory: SeriesFactory,
    name: str,
    data: list[Any],
    call: Callable[[Any], EagerSeries[Any]],
) -> None:
    series = factory(data)
    series._broadcast = True
    result = call(series)
    assert len(result) != 1, name
    assert result._broadcast is False, name


@pytest.mark.parametrize(
    ("name", "data", "call"), PRESERVING, ids=[case[0] for case in PRESERVING]
)
def test_never_gains_broadcast(
    factory: SeriesFactory,
    name: str,
    data: list[Any],
    call: Callable[[Any], EagerSeries[Any]],
) -> None:
    """A series which was not standing for a scalar must never start to."""
    series = factory(data * 3)
    assert series._broadcast is False
    assert call(series)._broadcast is False, name


def test_zip_with_broadcast_mask(factory: SeriesFactory) -> None:
    """A broadcast mask against a full-length series must not raise."""
    mask = factory([False])
    mask._broadcast = True
    series = factory(["a", "b", "c"])
    result = series.zip_with(mask, factory(["x", "y", "z"]))
    assert result.to_list() == ["x", "y", "z"]
    assert result._broadcast is False
