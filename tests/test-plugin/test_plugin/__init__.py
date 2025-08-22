from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import TypeIs

from narwhals.utils import Version

if TYPE_CHECKING:
    from test_plugin.dataframe import DictFrame, DictLazyFrame


def from_native(native_object: DictFrame) -> DictLazyFrame:
    from test_plugin.dataframe import DictLazyFrame

    return DictLazyFrame(native_object, version=Version.MAIN)


def is_native_object(obj: Any) -> TypeIs[DictFrame]:
    return isinstance(obj, dict)
