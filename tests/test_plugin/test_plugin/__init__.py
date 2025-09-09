from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import TypeIs

if TYPE_CHECKING:
    from test_plugin.dataframe import (  # type: ignore[import-untyped, import-not-found, unused-ignore]
        DictFrame,
    )

    from narwhals.utils import Version
    from tests.test_plugin.test_plugin.namespace import DictNamespace


# @mp: is the return type here correct?
def __narwhals_namespace__(version: Version) -> DictNamespace:  # noqa: N807
    from test_plugin.namespace import DictNamespace

    return DictNamespace(version=version)


def is_native_object(obj: Any) -> TypeIs[DictFrame]:
    return isinstance(obj, dict)
