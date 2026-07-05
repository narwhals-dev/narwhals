from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._utils import Implementation, not_implemented

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from narwhals._utils import Version, _LimitedContext
    from narwhals.series import Series


class DictSeries:
    """Minimal eager series, kept to the smallest surface exercised in narwhals' tests."""

    _implementation = Implementation.UNKNOWN

    def __init__(
        self, values: Iterable[Any], *, name: str = "", version: Version
    ) -> None:
        self._values: list[Any] = list(values)
        self._name = name
        self._version = version

    @classmethod
    def from_iterable(
        cls,
        data: Iterable[Any],
        /,
        *,
        context: _LimitedContext,
        name: str = "",
        dtype: Any = None,  # noqa: ARG003
    ) -> Self:
        return cls(data, name=name, version=context._version)

    @classmethod
    def from_numpy(cls, data: Any, /, *, context: _LimitedContext) -> Self:
        return cls(data.tolist(), version=context._version)

    def __narwhals_series__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> Any:
        from test_plugin.namespace import DictNamespace

        return DictNamespace(version=self._version)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def native(self) -> list[Any]:
        return self._values

    @property
    def name(self) -> str:
        return self._name

    def _with_version(self, version: Version) -> Self:
        return self.__class__(self._values, name=self._name, version=version)

    def alias(self, name: str) -> Self:
        return self.__class__(self._values, name=name, version=self._version)

    def is_empty(self) -> bool:
        return not self._values

    def scatter(self, indices: Self, values: Self) -> Self:
        data = list(self._values)
        for index, value in zip(indices.native, values.native, strict=True):
            data[index] = value
        return self.__class__(data, name=self._name, version=self._version)

    def to_narwhals(self) -> Series[Any]:
        return self._version.series(self, level="full")

    cast: Any = not_implemented()
