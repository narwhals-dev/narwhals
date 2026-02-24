from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING, Any, overload

from narwhals._utils import Implementation, Version, is_eager_allowed

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan import arrow as _arrow
    from narwhals._plan.compliant.dataframe import EagerDataFrame
    from narwhals._plan.compliant.namespace import CompliantNamespace, EagerNamespace
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.typing import NativeDataFrameT, NativeSeriesT
    from narwhals._typing import Arrow
    from narwhals.typing import Backend, EagerAllowed, IntoBackend

    # NOTE: Use when you have a function that calls a namespace method, and eventually returns:
    # - `DataFrame[NativeDataFrameT]`, or
    # - `Series[NativeSeriesT]`
    EagerNs: TypeAlias = EagerNamespace[
        EagerDataFrame[Any, NativeDataFrameT, Any],
        CompliantSeries[NativeSeriesT],
        Any,
        Any,
    ]


def namespace(backend: IntoBackend[Backend]) -> CompliantNamespace[Any, Any, Any]:
    impl = Implementation.from_backend(backend)
    if is_eager_allowed(impl):
        return eager_namespace(impl)
    msg = f"Lazy backends are not yet supported in `narwhals._plan`, got: {impl!r}"
    raise NotImplementedError(msg)


@overload
def eager_namespace(backend: Arrow, /) -> _arrow.Namespace: ...
@overload
def eager_namespace(backend: IntoBackend[EagerAllowed], /) -> EagerNs[Any, Any]: ...
def eager_namespace(
    backend: IntoBackend[EagerAllowed], /
) -> EagerNs[t.Any, t.Any] | _arrow.Namespace:
    impl = Implementation.from_backend(backend)
    if is_eager_allowed(impl):
        if impl is Implementation.PYARROW:
            from narwhals._plan import arrow as _arrow

            return _arrow.Namespace(Version.MAIN)
        raise NotImplementedError(impl)
    msg = f"{impl} support in Narwhals is lazy-only"
    raise ValueError(msg)
