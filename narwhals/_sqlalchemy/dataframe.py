from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Select
from sqlalchemy import Table
from sqlalchemy import select

from narwhals.utils import Implementation
from narwhals.utils import validate_backend_version

if TYPE_CHECKING:
    from types import ModuleType

    from typing_extensions import Self

    from narwhals._sqlalchemy.namespace import SQLAlchemyNamespace
    from narwhals._sqlalchemy.typing import IntoSQLAlchemyExpr
    from narwhals.utils import Version


class SQLAlchemyLazyFrame:
    def __init__(
        self: Self,
        native_dataframe: Select | Table,
        *,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._native_frame: Select = (
            select(native_dataframe)
            if isinstance(native_dataframe, Table)
            else native_dataframe
        )
        self._backend_version = backend_version
        self._implementation = Implementation.SQLALCHEMY
        self._version = version
        validate_backend_version(self._implementation, self._backend_version)

    def __native_namespace__(self: Self) -> ModuleType:  # pragma: no cover
        if self._implementation is Implementation.SQLALCHEMY:
            return self._implementation.to_native_namespace()

        msg = (
            f"Expected sqlalchemy, got: {type(self._implementation)}"  # pragma: no cover
        )
        raise AssertionError(msg)

    def __narwhals_namespace__(self: Self) -> SQLAlchemyNamespace:
        from narwhals._sqlalchemy.namespace import SQLAlchemyNamespace

        return SQLAlchemyNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def __narwhals_lazyframe__(self: Self) -> Self:
        return self

    def _from_native_frame(self: Self, df: Table) -> Self:
        return self.__class__(
            df, backend_version=self._backend_version, version=self._version
        )

    @property
    def columns(self: Self) -> list[str]:
        return self._native_frame.selected_columns.keys()  # type: ignore[no-any-return]

    def select(
        self: Self,
        *exprs: IntoSQLAlchemyExpr,
        **named_exprs: IntoSQLAlchemyExpr,
    ) -> Self:
        table = self._native_frame

        if all(isinstance(e, str) for e in exprs) and all(
            isinstance(e, str) for e in named_exprs.values()
        ):
            # fastpath
            return self._from_native_frame(
                select(
                    *[getattr(table.c, name) for name in exprs],  # type: ignore[arg-type]
                    *[
                        getattr(table.c, name).label(new_name)  # type: ignore[arg-type]
                        for new_name, name in named_exprs.items()
                    ],
                )
            )
        msg = "todo"
        raise NotImplementedError(msg)
