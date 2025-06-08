from __future__ import annotations

import sys
from importlib.util import find_spec
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from narwhals.typing import IntoFrameT

data = {"a": [1, 1, 2], "b": [4, 5, 6]}


def _roundtrip_query(frame: IntoFrameT) -> IntoFrameT:
    return (
        nw.from_native(frame)
        .group_by("a")
        .agg(nw.col("b").mean())
        .filter(nw.col("a") > 1)
        .to_native()
    )


@pytest.mark.parametrize(
    ("impl", "frame_constructor", "frame_type", "kwargs"),
    [
        (Implementation.CUDF, "DataFrame", "DataFrame", {}),
        (Implementation.DASK, "from_dict", "DataFrame", {"npartitions": 1}),
        (Implementation.MODIN, "DataFrame", "DataFrame", {}),
        (Implementation.PANDAS, "DataFrame", "DataFrame", {}),
        (Implementation.POLARS, "DataFrame", "DataFrame", {}),
        (Implementation.PYARROW, "table", "Table", {}),
    ],
)
def test_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    impl: Implementation,
    frame_constructor: str,
    frame_type: str,
    kwargs: dict[str, Any],
) -> None:
    module_name = impl.value
    if not find_spec(module_name):
        reason = f"{module_name} not installed"
        pytest.skip(reason=reason)

    if impl in {Implementation.DASK, Implementation.PYARROW} or (
        sys.version_info >= (3, 10) and impl is Implementation.PANDAS
    ):
        monkeypatch.delitem(sys.modules, module_name)

    module = impl.to_native_namespace()
    df = getattr(module, frame_constructor)(data, **kwargs)
    result = _roundtrip_query(df)

    assert isinstance(result, getattr(module, frame_type))


@pytest.mark.parametrize(
    "impl",
    [
        Implementation.CUDF,
        Implementation.DASK,
        Implementation.DUCKDB,
        Implementation.IBIS,
        Implementation.MODIN,
        Implementation.PANDAS,
        Implementation.POLARS,
        Implementation.PYARROW,
        Implementation.PYSPARK,
        Implementation.SQLFRAME,
    ],
)
def test_to_native_namespace(
    monkeypatch: pytest.MonkeyPatch, impl: Implementation
) -> None:
    if not find_spec(impl.value):
        reason = f"{impl.value} not installed"
        pytest.skip(reason=reason)

    assert isinstance(impl.to_native_namespace(), ModuleType)

    monkeypatch.setattr(
        "narwhals._utils.Implementation._backend_version", lambda _: (0, 0, 1)
    )

    with pytest.raises(ValueError, match="Minimum version"):
        impl.to_native_namespace()


def test_to_native_namespace_unknown() -> None:
    impl = Implementation.UNKNOWN
    with pytest.raises(
        AssertionError, match="Cannot return native namespace from UNKNOWN Implementation"
    ):
        impl.to_native_namespace()
