from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, assert_equal_series

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._typing import EagerAllowed
    from narwhals.typing import IntoDType
    from tests.utils import IntoIterable


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ((4, 1, 2), nw.Int32),
        ((-1, 5, 100), None),
        ((2.1, 2.7, 2.0), nw.Float64),
        (("one", "two"), nw.String),
    ],
    ids=["Int32", "no-dtype", "Float64", "String"],
)
def test_series_from_iterable(
    eager_implementation: EagerAllowed,
    into_iter_16: IntoIterable,
    values: Sequence[Any],
    dtype: IntoDType,
    request: pytest.FixtureRequest,
) -> None:
    name = "b"
    iterable = into_iter_16(values)
    test_name = request.node.name
    request.applymarker(
        pytest.mark.xfail(
            ("polars-pandas" in test_name and "array" in test_name),
            raises=TypeError,
            reason="Polars doesn't support `pd.array`.\nhttps://github.com/pola-rs/polars/issues/22757",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            (
                "pandas-polars" in test_name
                and "String" in test_name
                and PANDAS_VERSION >= (3,)
            ),
            reason=(
                "Pandas nightly suddenly raising on String `pl.Series` in:\n"
                "https://github.com/pandas-dev/pandas/blob/3ea783ea21e22035cf0a3605cfde3178e9348ee1/pandas/core/arrays/string_arrow.py#L202-L204"
            ),
        )
    )
    if (
        "pandas-pyarrow" in test_name
        and "array-String" in test_name
        and PANDAS_VERSION < (2, 1)
    ):  # pragma: no cover
        pytest.skip(
            "pandas being pandas with strings https://github.com/narwhals-dev/narwhals/pull/2933#issuecomment-3156009516"
        )
    result = nw.Series.from_iterable(name, iterable, dtype, backend=eager_implementation)
    if dtype:
        assert result.dtype == dtype
    assert_equal_series(result, values, name)


@pytest.mark.parametrize(("values", "expected_dtype"), [((4, 1, 2), nw.Int64)])
def test_series_from_iterable_infer(
    eager_backend: EagerAllowed, values: Sequence[Any], expected_dtype: IntoDType
) -> None:
    name = "b"
    result = nw.Series.from_iterable(name, values, backend=eager_backend)
    assert result.dtype == expected_dtype
    assert_equal_series(result, values, name)


def test_series_from_iterable_not_eager() -> None:
    backend = "sqlframe"
    pytest.importorskip(backend)
    with pytest.raises(ValueError, match="lazy-only"):
        nw.Series.from_iterable("", [1, 2, 3], backend=backend)  # type: ignore[arg-type]


def test_series_from_iterable_numpy_not_1d(eager_backend: EagerAllowed) -> None:
    pytest.importorskip("numpy")
    import numpy as np

    with pytest.raises(ValueError, match=r"only.+1D numpy arrays"):
        nw.Series.from_iterable("", np.array([[0], [2]]), backend=eager_backend)


def test_series_from_iterable_not_iterable(eager_backend: EagerAllowed) -> None:
    with pytest.raises(TypeError, match=r"iterable.+got.+int"):
        nw.Series.from_iterable("", 2000, backend=eager_backend)  # type: ignore[arg-type]
