from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterable

from narwhals.schema import Schema as NwSchema
from narwhals.utils import Version
from narwhals.utils import inherit_doc

if TYPE_CHECKING:
    from typing import Mapping

    from narwhals.stable.v1.dtypes import DType


class Schema(NwSchema):
    _version = Version.V1

    @inherit_doc(NwSchema)
    def __init__(
        self, schema: Mapping[str, DType] | Iterable[tuple[str, DType]] | None = None
    ) -> None:
        super().__init__(schema)
