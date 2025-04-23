from __future__ import annotations

from typing import Any
from typing import TypeVar

from narwhals import DataFrame
from narwhals import LazyFrame

FrameT = TypeVar("FrameT", DataFrame[Any], LazyFrame[Any])
