from __future__ import annotations

from queries import q20

from . import IO_FUNCS
from . import lineitem
from . import nation
from . import part
from . import partsupp
from . import supplier

fn = IO_FUNCS["pandas[pyarrow]"]
print(q20.query(fn(part), fn(partsupp), fn(nation), fn(lineitem), fn(supplier)))

fn = IO_FUNCS["polars[lazy]"]
print(q20.query(fn(part), fn(partsupp), fn(nation), fn(lineitem), fn(supplier)).collect())

fn = IO_FUNCS["pyarrow"]
print(q20.query(fn(part), fn(partsupp), fn(nation), fn(lineitem), fn(supplier)))

fn = IO_FUNCS["dask"]
print(q20.query(fn(part), fn(partsupp), fn(nation), fn(lineitem), fn(supplier)).compute())
