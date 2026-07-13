"""Experimenting with how plugins fit into `{Eager,Lazy}{Allowed,Only}`.

## Notes
- [`tests.plan.plugin_test`] covers similar typing concerns
    - for [`Plugin`][narwhals._plan.plugins.Plugin] and [`Builtin`][narwhals._plan.plugins.Builtin]
    - the classes they wrap
- This guy is specifically ([#3753]), where the typing (here) hasn't been updated for to match the runtime


[`tests.plan.plugin_test`]: https://github.com/narwhals-dev/narwhals/blob/17fbd0fae85d6436b4619d44d273a6dcca30a8ee/tests/plan/plugins_test.py

[#3753]: https://github.com/narwhals-dev/narwhals/pull/3753
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

import pytest

import narwhals._plan as nwp


# NOTE: We want this to continue warning
def test_series_from_iterable_lazy_only() -> None:
    backend = "duckdb"
    pytest.importorskip(backend)
    with pytest.raises(
        NotImplementedError, match=re.escape("`Series()` is not supported for 'duckdb'")
    ):
        nwp.Series.from_iterable([1, 2, 3], backend=backend)  # type: ignore[arg-type]


# NOTE: We want this to continue warning
def test_dataframe_from_dict_lazy_only() -> None:
    backend = "duckdb"
    pytest.importorskip(backend)
    with pytest.raises(
        NotImplementedError,
        match=re.escape("`DataFrame()` is not supported for 'duckdb'"),
    ):
        nwp.DataFrame.from_dict({"a": (1, 2), "b": (3, 4)}, backend=backend)  # type: ignore[call-overload]


# TODO @dangotbanned: All of this requires updating the source
if TYPE_CHECKING:
    from narwhals._plan.typing import IntoPlugin, PluginName
    from narwhals._utils import Implementation
    from narwhals.typing import EagerAllowed, IntoBackend, LazyAllowed

    # TODO @dangotbanned: Introduce `PluginName` to `backend`
    # TODO @dangotbanned: Add overloads like all the other constructors
    def typing_series_from_iterable(
        plugin_only: PluginName,
        eager_allowed: EagerAllowed,
        eager_allowed_or_plugin: EagerAllowed | PluginName,
        into_eager_allowed: IntoBackend[EagerAllowed],
        into_eager_allowed_or_plugin: IntoBackend[EagerAllowed] | PluginName,
        lazy_allowed: LazyAllowed,
        lazy_allowed_or_plugin: LazyAllowed | PluginName,
        into_lazy_allowed: IntoBackend[LazyAllowed],
        into_lazy_allowed_or_plugin: IntoBackend[LazyAllowed] | PluginName,
        into_plugin: IntoPlugin,
        dynamic_string: str,
        disjoint_type: int,
        implementation_opaque: Implementation,
        implementation_unknown: Literal[Implementation.UNKNOWN],
        fully_unknown: Any,
    ) -> None:
        v = [1, 2, 3]
        n = "name"

        nwp.Series.from_iterable(name=n, values=v, backend=plugin_only)  # type: ignore[arg-type]
        nwp.Series.from_iterable(name=n, values=v, backend=eager_allowed)
        nwp.Series.from_iterable(name=n, values=v, backend=eager_allowed_or_plugin)  # type: ignore[arg-type]
        nwp.Series.from_iterable(name=n, values=v, backend=into_eager_allowed)
        nwp.Series.from_iterable(name=n, values=v, backend=into_eager_allowed_or_plugin)  # type: ignore[arg-type]
        nwp.Series.from_iterable(name=n, values=v, backend=lazy_allowed)  # type: ignore[arg-type]
        nwp.Series.from_iterable(name=n, values=v, backend=lazy_allowed_or_plugin)  # type: ignore[arg-type]
        nwp.Series.from_iterable(name=n, values=v, backend=into_lazy_allowed)  # type: ignore[arg-type]
        nwp.Series.from_iterable(name=n, values=v, backend=into_lazy_allowed_or_plugin)  # type: ignore[arg-type]
        nwp.Series.from_iterable(name=n, values=v, backend=into_plugin)  # type: ignore[arg-type]
        nwp.Series.from_iterable(name=n, values=v, backend=dynamic_string)  # type: ignore[arg-type]
        nwp.Series.from_iterable(name=n, values=v, backend=disjoint_type)  # type: ignore[arg-type]
        nwp.Series.from_iterable(name=n, values=v, backend=implementation_opaque)  # type: ignore[arg-type]
        nwp.Series.from_iterable(name=n, values=v, backend=implementation_unknown)  # type: ignore[arg-type]
        nwp.Series.from_iterable(name=n, values=v, backend=fully_unknown)

    # TODO @dangotbanned: Introduce `PluginName` to `backend`
    def typing_dataframe_from_dict(
        plugin_only: PluginName,
        eager_allowed: EagerAllowed,
        eager_allowed_or_plugin: EagerAllowed | PluginName,
        into_eager_allowed: IntoBackend[EagerAllowed],
        into_eager_allowed_or_plugin: IntoBackend[EagerAllowed] | PluginName,
        lazy_allowed: LazyAllowed,
        lazy_allowed_or_plugin: LazyAllowed | PluginName,
        into_lazy_allowed: IntoBackend[LazyAllowed],
        into_lazy_allowed_or_plugin: IntoBackend[LazyAllowed] | PluginName,
        into_plugin: IntoPlugin,
        dynamic_string: str,
        disjoint_type: int,
        implementation_opaque: Implementation,
        implementation_unknown: Literal[Implementation.UNKNOWN],
        fully_unknown: Any,
    ) -> None:
        data: dict[str, tuple[int, int]] = {"a": (1, 2), "b": (3, 4)}

        nwp.DataFrame.from_dict(data, backend=plugin_only)  # type: ignore[call-overload]
        nwp.DataFrame.from_dict(data, backend=eager_allowed)
        nwp.DataFrame.from_dict(data, backend=eager_allowed_or_plugin)  # type: ignore[arg-type]
        nwp.DataFrame.from_dict(data, backend=into_eager_allowed)
        nwp.DataFrame.from_dict(data, backend=into_eager_allowed_or_plugin)  # type: ignore[arg-type]
        nwp.DataFrame.from_dict(data, backend=lazy_allowed)  # type: ignore[arg-type]
        nwp.DataFrame.from_dict(data, backend=lazy_allowed_or_plugin)  # type: ignore[arg-type]
        nwp.DataFrame.from_dict(data, backend=into_lazy_allowed)  # type: ignore[arg-type]
        nwp.DataFrame.from_dict(data, backend=into_lazy_allowed_or_plugin)  # type: ignore[arg-type]
        nwp.DataFrame.from_dict(data, backend=into_plugin)  # type: ignore[arg-type]
        nwp.DataFrame.from_dict(data, backend=dynamic_string)  # type: ignore[call-overload]
        nwp.DataFrame.from_dict(data, backend=disjoint_type)  # type: ignore[call-overload]
        nwp.DataFrame.from_dict(data, backend=implementation_opaque)  # type: ignore[call-overload]
        nwp.DataFrame.from_dict(data, backend=implementation_unknown)  # type: ignore[call-overload]
        nwp.DataFrame.from_dict(data, backend=fully_unknown)
