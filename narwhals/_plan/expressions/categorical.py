from __future__ import annotations

from typing import ClassVar

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._function import Function
from narwhals._plan.expressions.namespace import IRNamespace


# fmt: off
class CategoricalFunction(Function, dispatch=DispatcherOptions(accessor_name="cat")): ...
class GetCategories(CategoricalFunction, dtype=dtm.STR): ...
# fmt: on
class IRCatNamespace(IRNamespace):
    get_categories: ClassVar = GetCategories
