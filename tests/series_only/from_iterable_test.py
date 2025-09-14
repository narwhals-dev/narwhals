from __future__ import annotations

from collections import deque
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._utils import qualified_type_name
from tests.utils import PANDAS_VERSION, assert_equal_series

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
        Iterator,
        KeysView,
        Sequence,
        ValuesView,
    )

    from typing_extensions import TypeAlias

    from narwhals._typing import EagerAllowed
    from narwhals.typing import IntoDType

    IntoIterable: TypeAlias = Callable[..., Iterable[Any]]


class UserDefinedIterable:
    def __init__(self, iterable: Iterable[Any]) -> None:
        self.iterable: Iterable[Any] = iterable

    def __iter__(self) -> Iterator[Any]:
        yield from self.iterable


def generator_function(iterable: Iterable[Any]) -> Generator[Any, Any, None]:
    yield from iterable


def generator_expression(iterable: Iterable[Any]) -> Generator[Any, None, None]:
    return (element for element in iterable)


def dict_keys(iterable: Iterable[Any]) -> KeysView[Any]:
    return dict.fromkeys(iterable).keys()


def dict_values(iterable: Iterable[Any]) -> ValuesView[Any]:
    return dict(enumerate(iterable)).values()


_INTO_ITER_3RD_PARTY: list[IntoIterable] = []

if find_spec("numpy"):  # pragma: no cover
    import numpy as np

    _INTO_ITER_3RD_PARTY.append(np.array)
else:  # pragma: no cover
    ...
if find_spec("pandas"):  # pragma: no cover
    import pandas as pd

    _INTO_ITER_3RD_PARTY.extend([pd.Index, pd.array, pd.Series])
else:  # pragma: no cover
    ...
if find_spec("polars"):  # pragma: no cover
    import polars as pl

    _INTO_ITER_3RD_PARTY.append(pl.Series)
else:  # pragma: no cover
    ...
if find_spec("pyarrow"):  # pragma: no cover
    import pyarrow as pa

    def chunked_array(iterable: Any) -> Iterable[Any]:
        return pa.chunked_array([iterable])

    _INTO_ITER_3RD_PARTY.extend([pa.array, chunked_array])
else:  # pragma: no cover
    ...

_INTO_ITER_STDLIB: tuple[IntoIterable, ...] = (
    list,
    tuple,
    iter,
    deque,
    generator_function,
    generator_expression,
)
_INTO_ITER_STDLIB_EXOTIC: tuple[IntoIterable, ...] = dict_keys, dict_values
INTO_ITER: tuple[IntoIterable, ...] = (
    *_INTO_ITER_STDLIB,
    *_INTO_ITER_STDLIB_EXOTIC,
    UserDefinedIterable,
    *_INTO_ITER_3RD_PARTY,
)


def _ids_into_iter(obj: Any) -> str:
    module: str = ""
    if (obj_module := obj.__module__) and obj_module != __name__:
        module = obj.__module__
    name = qualified_type_name(obj)
    if name in {"function", "builtin_function_or_method"} or "_cython" in name:
        return f"{module}.{obj.__qualname__}" if module else obj.__qualname__
    return name.removeprefix(__name__).strip(".")


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
@pytest.mark.parametrize("into_iter", INTO_ITER, ids=_ids_into_iter)
def test_series_from_iterable(
    eager_implementation: EagerAllowed,
    values: Sequence[Any],
    dtype: IntoDType,
    into_iter: IntoIterable,
    request: pytest.FixtureRequest,
) -> None:
    name = "b"
    iterable = into_iter(values)
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
