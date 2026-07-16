# ruff: noqa: ARG003
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import numpy as np
import pandas as pd

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self


# `cast`-preserving sparse support is best-effort.
# pandas<2.0 has different sparse `astype` behaviour.
# Schema *reading* of numeric/bool subtypes works everywhere.
REQUIRES_PANDAS_V2 = pytest.mark.skipif(
    PANDAS_VERSION < (2,), reason="sparse cast support targets pandas>=2.0"
)


class CustomInt16Dtype(pd.api.extensions.ExtensionDtype):  # pragma: no cover
    name = "custom_int_16"
    type = int
    _metadata = ("name",)

    @classmethod
    def construct_array_type(cls) -> type[pd.api.extensions.ExtensionArray]:  # type: ignore[valid-type]
        return CustomInt16Array  # pyrefly: ignore[bad-return]


class CustomInt16Array(pd.api.extensions.ExtensionArray):  # pragma: no cover
    def __init__(self, values: Sequence[Any]) -> None:
        self._values = pd.array(values, dtype="int16")

    @classmethod
    def _from_sequence(
        cls, scalars: Sequence[Any], *, dtype: Any = None, copy: bool = False
    ) -> Self:
        return cls(scalars)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def dtype(self) -> pd.api.extensions.ExtensionDtype:
        return CustomInt16Dtype()

    def copy(self) -> Self:
        return type(self)(list(self._values.copy()))


class CustomInt32Dtype(pd.api.extensions.ExtensionDtype):  # pragma: no cover
    name = "custom_int_32"
    type = int
    _metadata = ("name",)

    @classmethod
    def construct_array_type(cls) -> type[pd.api.extensions.ExtensionArray]:  # type: ignore[valid-type]
        return CustomInt32Array  # pyrefly: ignore[bad-return]

    def __hash__(self) -> int:
        return hash(self.name)


class CustomInt32Array(pd.api.extensions.ExtensionArray):  # pragma: no cover
    def __init__(self, values: Sequence[Any]) -> None:
        self._values = pd.array(values, dtype="int32")

    @classmethod
    def _from_sequence(
        cls, scalars: Sequence[Any], *, dtype: Any = None, copy: bool = False
    ) -> Self:
        return cls(scalars)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def dtype(self) -> pd.api.extensions.ExtensionDtype:
        return CustomInt16Dtype()

    def copy(self) -> Self:
        return type(self)(list(self._values.copy()))


def test_dataframe_with_ext() -> None:
    int16_array = CustomInt16Array([1, 2])
    int32_array = CustomInt32Array([1, 2])

    df = pd.DataFrame({"non-hash-int16": int16_array, "hash-int-32": int32_array})

    assert nw.from_native(df).schema == {
        "non-hash-int16": nw.Unknown(),
        "hash-int-32": nw.Unknown(),
    }


def test_schema_with_ext() -> None:
    pd_schema = {"non-hash-int16": CustomInt16Dtype(), "hash-int-32": CustomInt32Dtype()}
    nw_schema = nw.Schema.from_pandas_like(pd_schema)
    assert nw_schema == nw.Schema(
        {"non-hash-int16": nw.Unknown(), "hash-int-32": nw.Unknown()}
    )


def test_sparse_schema() -> None:
    # See https://github.com/narwhals-dev/narwhals/issues/3722.
    # Sparse columns map to the narwhals dtype of their dense subtype (not `Unknown`).
    # The string/object subtype needs sample densification (which pre-2.0 pandas doesn't
    # do), so it's covered separately by `test_sparse_object_cast_string_densifies`.
    df = pd.DataFrame(
        {
            "a": pd.arrays.SparseArray([0, 1, 0, 2]),
            "b": pd.arrays.SparseArray([0.0, 1.5, 0.0, 2.5]),
            "c": pd.arrays.SparseArray([True, False, False, True]),
        }
    )
    assert nw.from_native(df).schema == nw.Schema(
        {"a": nw.Int64(), "b": nw.Float64(), "c": nw.Boolean()}
    )
    assert nw.from_native(df["a"], series_only=True).dtype == nw.Int64


@REQUIRES_PANDAS_V2
def test_sparse_cast_preserves_sparsity() -> None:
    # Casting a sparse column to a numpy-subtype target keeps it sparse *and* must not
    # corrupt the data: a naive `astype(SparseDtype("float64"))` keeps the sparse index
    # but resets the fill to `NaN`, turning every compressed `0` into `NaN`.
    s = nw.from_native(
        pd.Series(pd.arrays.SparseArray([0, 0, 1, 0, 2])), series_only=True
    )
    result = s.cast(nw.Float64)
    native = nw.to_native(result)
    assert isinstance(native.dtype, pd.SparseDtype)
    # `pandas-stubs`' `SparseDtype` doesn't expose `.subtype`, so assert the densified
    # dtype instead (which also confirms the values round-trip correctly).
    arr = native.to_numpy()
    assert arr.dtype == "float64"
    assert arr.tolist() == [0.0, 0.0, 1.0, 0.0, 2.0]

    # Same logical dtype: stays sparse (no densification).
    assert isinstance(nw.to_native(s.cast(nw.Int64)).dtype, pd.SparseDtype)


@REQUIRES_PANDAS_V2
@pytest.mark.parametrize(
    ("data", "target"),
    [
        # NaN fill can't be represented as an integer subtype -> densify.
        (pd.arrays.SparseArray([0.0, 1.0, 2.0]), nw.Int64),
        # String / object subtypes densify.
        (pd.arrays.SparseArray([0, 1, 2]), nw.String),
        # `SparseDtype` can only wrap numpy dtypes, so non-numpy targets densify.
        (pd.arrays.SparseArray([0, 1, 2]), nw.Categorical),
    ],
)
def test_sparse_cast_densifies_when_unsupported(
    data: Any, target: nw.dtypes.DType
) -> None:
    s = nw.from_native(pd.Series(data), series_only=True)
    native = nw.to_native(s.cast(target))
    assert not isinstance(native.dtype, pd.SparseDtype)


@REQUIRES_PANDAS_V2
@pytest.mark.parametrize(
    ("data", "target"),
    [
        # Non-zero fill carried across to a wider numeric subtype: the fill (7) must
        # survive as 7.0, not reset to the float default (NaN).
        (pd.arrays.SparseArray([7, 7, 1, 7, 2], fill_value=7), nw.Float64),
        # Bool subtype (`kind == "b"`) stays sparse.
        (pd.arrays.SparseArray([True, False, False, True]), nw.Int64),
        # Datetime subtype (`kind == "M"`); source unit differs from target to force
        # the slow path through `keep_sparse_dtype`.
        (
            pd.arrays.SparseArray(
                ["2020-01-01", "2020-01-01", "2020-01-02"], dtype="datetime64[ms]"
            ),
            nw.Datetime("us"),
        ),
        # Timedelta subtype (`kind == "m"`); source unit differs from target.
        (pd.arrays.SparseArray([0, 0, 5], dtype="timedelta64[us]"), nw.Duration("ns")),
    ],
)
def test_sparse_cast_stays_sparse(data: Any, target: nw.dtypes.DType) -> None:
    # A sparse cast must equal the dense cast, only kept sparse. Comparing the two
    # densified results proves the fill was carried (a reset fill would corrupt the
    # compressed entries) without hard-coding per-dtype expected values.
    sparse_native = pd.Series(data)
    dense_native = pd.Series(np.asarray(data))

    sparse_result = nw.to_native(
        nw.from_native(sparse_native, series_only=True).cast(target)
    )
    dense_result = nw.to_native(
        nw.from_native(dense_native, series_only=True).cast(target)
    )

    assert isinstance(sparse_result.dtype, pd.SparseDtype)
    assert not isinstance(dense_result.dtype, pd.SparseDtype)
    np.testing.assert_array_equal(sparse_result.to_numpy(), dense_result.to_numpy())


@REQUIRES_PANDAS_V2
def test_sparse_object_cast_string_densifies() -> None:
    # A sparse `object` column sniffs as `String`, so casting to `String` is a no-op at
    # the narwhals level. The fast path in `cast` must NOT keep it sparse: an
    # `astype(str)` on a sparse `object` column does not actually stringify the values.
    s = nw.from_native(
        pd.Series(pd.arrays.SparseArray(["x", "y", "x", "z"])), series_only=True
    )
    assert s.dtype == nw.String
    native = nw.to_native(s.cast(nw.String))
    assert not isinstance(native.dtype, pd.SparseDtype)
