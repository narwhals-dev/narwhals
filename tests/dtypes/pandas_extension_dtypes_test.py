# ruff: noqa: ARG003
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("pandas")


import pandas as pd

import narwhals as nw

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self


class CustomInt16Dtype(pd.api.extensions.ExtensionDtype):  # pragma: no cover
    name = "custom_int_16"
    type = int
    _metadata = ("name",)

    @classmethod
    def construct_array_type(cls) -> type[pd.api.extensions.ExtensionArray]:  # type: ignore[valid-type]
        return CustomInt16Array


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
        return CustomInt32Array

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
