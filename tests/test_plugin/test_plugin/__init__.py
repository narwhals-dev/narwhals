from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narwhals.utils import Version
    from tests.test_plugin.test_plugin.dataframe import DictFrame as DictFrame
    from tests.test_plugin.test_plugin.namespace import DictNamespace


def __narwhals_namespace__(version: Version) -> DictNamespace:  # noqa: N807
    from tests.test_plugin.test_plugin.namespace import DictNamespace

    return DictNamespace(version=version)


NATIVE_PACKAGE = "builtins"
