"""Special module without future annotations for executing Python code."""

from typing import Any, Dict, Optional


def exec_python(code: str, filename: str, exec_globals: Optional[Dict[str, Any]] = None) -> None:
    compiled = compile(code, filename=filename, mode="exec")
    exec(compiled, exec_globals)  # noqa: S102
