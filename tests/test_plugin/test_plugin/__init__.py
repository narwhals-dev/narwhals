from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import TypeIs

if TYPE_CHECKING:
    from test_plugin.dataframe import (  # type: ignore[import-untyped, import-not-found, unused-ignore]
        DictFrame,
        DictLazyFrame,
    )

    from narwhals.utils import Version


def from_native(native_object: DictFrame, version: Version) -> DictLazyFrame:
    from test_plugin.dataframe import DictLazyFrame

    return DictLazyFrame(native_object, version=version)


def is_native_object(obj: Any) -> TypeIs[DictFrame]:
    return isinstance(obj, dict)
