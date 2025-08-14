from __future__ import annotations

import daft

import narwhals as nw

nw.from_native(daft.from_pydict({"a": [1, 2, 3]}))
