from typing import Literal
from typing import get_args

API_VERSION = Literal["0.20"]
DEFAULT_API_VERSION: API_VERSION = "0.20"


def validate_api_version(version: API_VERSION) -> None:
    if version not in (args := get_args(API_VERSION)):
        msg = f"Expected one of {args}, got {version}"
        raise ValueError(msg)
