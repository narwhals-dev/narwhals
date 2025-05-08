from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile

wheel_path = next(Path("dist").glob("*.whl"))

with ZipFile(wheel_path) as z:
    # Allow only 'narwhals' and 'narwhals-<version>.dist-info' (metadata)
    unexpected_dirs = {
        dir_name
        for name in z.namelist()
        if not (dir_name := name.split("/")[0]).startswith("narwhals")
    }

    if unexpected_dirs:
        print(f"🚨 Unexpected directories in wheel: {unexpected_dirs}")  # noqa: T201
        sys.exit(1)

    sys.exit(0)
