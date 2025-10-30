from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._compliant import CompliantNamespace
from narwhals._utils import not_implemented
from tests.test_plugin.test_plugin.dataframe import DictFrame, DictLazyFrame

if TYPE_CHECKING:
    from narwhals.utils import Version


class DictNamespace(CompliantNamespace[DictLazyFrame, Any]):
    def __init__(self, *, version: Version) -> None:
        self._version = version

    def from_native(self, native_object: DictFrame) -> DictLazyFrame:
        return DictLazyFrame(native_object, version=self._version)

    is_native: Any = not_implemented()
    _expr: Any = not_implemented()
    _implementation: Any = not_implemented()
    len: Any = not_implemented()
    lit: Any = not_implemented()
    all_horizontal: Any = not_implemented()
    any_horizontal: Any = not_implemented()
    sum_horizontal: Any = not_implemented()
    mean_horizontal: Any = not_implemented()
    min_horizontal: Any = not_implemented()
    max_horizontal: Any = not_implemented()
    concat: Any = not_implemented()
    when: Any = not_implemented()
    concat_str: Any = not_implemented()
    selectors: Any = not_implemented()
    coalesce: Any = not_implemented()
