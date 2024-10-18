from __future__ import annotations

from queries import q19

from . import IO_FUNCS
from . import lineitem
from . import part

fn = IO_FUNCS["pandas[pyarrow]"]
print(q19.query(fn(lineitem), fn(part)))

fn = IO_FUNCS["polars[lazy]"]
print(q19.query(fn(lineitem), fn(part)).collect())

fn = IO_FUNCS["pyarrow"]
print(q19.query(fn(lineitem), fn(part)))

fn = IO_FUNCS["dask"]
print(q19.query(fn(lineitem), fn(part)).compute())
