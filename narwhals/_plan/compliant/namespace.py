from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol

from narwhals._plan.compliant import io, typing as ct
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from narwhals._utils import Implementation, Version


class CompliantNamespace(io.ReadSchema, Protocol[ct.E, ct.SC]):
    """`[ExprT_co, ScalarT_co]`."""

    __slots__ = ()

    implementation: ClassVar[Implementation]
    version: ClassVar[Version]

    @property
    def _expr(self) -> type[ct.E]: ...
    @property
    def _scalar(self) -> type[ct.SC]: ...

    # NOTE: will reduce direct calls to `*Namespace._<compliant-type>`
    from_native: not_implemented = not_implemented()


class EagerNamespace(CompliantNamespace[ct.E, ct.SC], Protocol[ct.DF, ct.S, ct.E, ct.SC]):
    """`[DataFrameT_co, SeriesT_co, ExprT_co, ScalarT_co]`."""

    __slots__ = ()

    @property
    def _series(self) -> type[ct.S]: ...
    @property
    def _dataframe(self) -> type[ct.DF]: ...
