from __future__ import annotations

import sys
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

dist_path = Path("dist")
wheel_path = next(dist_path.glob("*.whl"))
sdist_path = next(dist_path.glob("*.tar.gz"))

with ZipFile(wheel_path) as wheel_file:
    # Allow only 'narwhals' and 'narwhals-<version>.dist-info' (metadata)
    unexpected_wheel_dirs = {
        dir_name
        for name in wheel_file.namelist()
        if not (dir_name := name.split("/")[0]).startswith("narwhals")
    }

    if unexpected_wheel_dirs:
        print(f"🚨 Unexpected directories in wheel: {unexpected_wheel_dirs}")  # noqa: T201
        sys.exit(1)

with TarFile.open(sdist_path, mode="r:gz") as sdist_file:
    # Allow only 'narwhals' ans 'tests' folders
    sdist_dirs = {m.name.split("/")[1] for m in sdist_file.getmembers()}
    allowed_sdist_dirs = {
        "narwhals",
        "tests",
        "pyproject.toml",
        "PKG-INFO",
        "LICENSE.md",
        "README.md",
        ".gitignore",
    }

    if unexpected_sdist_dirs := sdist_dirs - allowed_sdist_dirs:
        print(f"🚨 Unexpected directories or files in sdist: {unexpected_sdist_dirs}")  # noqa: T201
        sys.exit(1)

sys.exit(0)
