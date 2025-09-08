from __future__ import annotations

from typing import TYPE_CHECKING, Any

from test_plugin.dataframe import DictFrame, DictLazyFrame

from narwhals._compliant import CompliantNamespace

if TYPE_CHECKING:
    from narwhals.utils import Version


class DictNamespace(CompliantNamespace[DictLazyFrame, Any]):
    def __init__(self, *, version: Version) -> None:
        self._version = version

    def from_native(self, native_object: DictFrame) -> DictLazyFrame:
        return DictLazyFrame(native_object, version=self._version)
