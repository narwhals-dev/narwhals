from __future__ import annotations

from pathlib import Path
from typing import Final

from narwhals.this import ZEN

DESTINATION_PATH: Final[Path] = Path("docs") / "this.md"

content = f"""
# The Zen of Narwhals

The well famous Python easter egg `import this` will reveal The Zen of Python (PEP 20).

Narwhals took inspiration from _this_ and created its own Zen.

```py
import narwhals.this
```

```terminal
{ZEN}
```
"""

with DESTINATION_PATH.open(mode="w") as destination:
    destination.write(content)
