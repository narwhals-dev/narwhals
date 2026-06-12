from __future__ import annotations

from typing import ClassVar

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan import _function as _f
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan.expressions.namespace import IRNamespace


# fmt: off
class CategoricalFunction(_f.Function, dispatch=DispatcherOptions(accessor_name="cat")): ...
class GetCategories(_f.UnaryFunction, CategoricalFunction, dtype=dtm.STR): ...
# fmt: on
class IRCatNamespace(IRNamespace):
    get_categories: ClassVar = GetCategories
