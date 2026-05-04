"""Protocols defining top-level constructor functions.

Their implementations delegate to other constructors that are classmethod(s)
on one or more types.

The delegation is not specified here, with the benefit being that you can make calls like this:

    namespace(obj).from_dict(...)
    namespace(obj).from_native(...)


And you would only need to check `obj` supports a single-method protocol:

    ns = namespace(obj)
    if can_from_dict(ns):
        return ns.from_dict(...)
    else:
        raise NotImplementedError

But you can also type this in a more plugin-friendly way:


    def from_dict(
        data: Mapping[str, Any],
        *,
        backend: IntoBackend[EagerAllowed] | FromDict[NativeDataFrameT, NativeSeriesT_co],
        schema: IntoSchema | None = None,
    ) -> (
        DataFrame[NativeDataFrameT, NativeSeriesT_co]  # <- We have the types already via `FromDict`
        | DataFrame[NativeDataFrame, NativeSeries]     # <- Internal backends can get overloads to improve this
    ): ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan.typing import NativeDataFrameT, NativeDataFrameT_co, NativeSeriesT_co
from narwhals._utils import _hasattr_static

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import TypeIs

    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals.typing import IntoSchema

__all__ = ("FromDict", "can_from_dict")


class FromDict(Protocol[NativeDataFrameT_co, NativeSeriesT_co]):
    """Namespace-level instance method, for initializing a dataframe.

    `[NativeDataFrameT_co, NativeSeriesT_co]`.
    """

    def from_dict(
        self, data: Mapping[str, Any], /, *, schema: IntoSchema | None = None
    ) -> CompliantDataFrame[NativeDataFrameT_co, NativeSeriesT_co]: ...


def can_from_dict(
    obj: FromDict[NativeDataFrameT, NativeSeriesT_co] | Any,
) -> TypeIs[FromDict[NativeDataFrameT, NativeSeriesT_co]]:
    return _hasattr_static(obj, "from_dict")
