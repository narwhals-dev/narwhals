from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import TypeIs
#from narwhals.utils import Version

if TYPE_CHECKING:
    from test_plugin.dataframe import (  # type: ignore[import-untyped, import-not-found, unused-ignore]
        DictFrame,
        DictLazyFrame,
    )
    from test_plugin.namespace import DictNamespace

    from narwhals.utils import Version


def __narwhals_namespace__(version: Version):
    from test_plugin.namespace import DictNamespace
    return DictNamespace(version=version)
    
def is_native_object(obj: Any) -> TypeIs[DictFrame]:
    return isinstance(obj, dict)
