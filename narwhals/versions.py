from __future__ import annotations

from typing import get_args

from narwhals.typing import API_VERSION

DEFAULT_API_VERSION: API_VERSION = "0.20"


def validate_api_version(version: API_VERSION) -> None:
    if version not in (args := get_args(API_VERSION)):
        msg = f"Expected one of {args}, got {version}"
        raise ValueError(msg)
