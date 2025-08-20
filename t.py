from __future__ import annotations

import daft

import narwhals as nw

df = nw.from_native(daft.from_pydict({"a": [1, 2, 3]}))

print(df)
print(type(df))
