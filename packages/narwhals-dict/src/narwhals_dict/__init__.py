from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals_dict.utils import is_native_frame

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._utils import Version
    from narwhals_dict.namespace import DictNamespace
    from narwhals_dict.typing import DictFrame

NATIVE_PACKAGE = "builtins"


def __narwhals_namespace__(version: Version) -> DictNamespace:  # noqa: N807
    from narwhals_dict.namespace import DictNamespace

    return DictNamespace(version=version)


def is_native(native_object: object) -> TypeIs[DictFrame]:
    return is_native_frame(native_object)
